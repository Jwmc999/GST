# Copyright 2020 The PEGASUS Authors..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Beam search.

This beam search implementation is designed for TPU usage only and prefers
flexibility over efficiency. Transformer attention caching is not enabled yet.

Mostly follows implementation in T2T. Several difference to pure beamsearch:
1. has finished and alive seqs, use 2 * beam_size to grow alive seqs,
   which makes beam_size=1 doesn't equal greedy.
2. prefers finished seq over alive seqs.
3. prefers lower indices when equal probability (though unlikely).
4. with custom length normalization and constraint.

Notations:
  B: batch_size, M: beam_size, T: max_decode_len, V: vocab_size, U: undefined
"""
# 
# pylint: disable=invalid-name

import torch

def length_normalization(start, alpha, min_len, max_len, out_of_range_penalty):
  r"""Create length normalization function.

  Combines length penalty from https://arxiv.org/abs/1609.08144,
  and length constraint from https://www.aclweb.org/anthology/W18-2706.pdf.

  scores = \sum_j log(P_j) / ((start + lengths)/(1 + start))**alpha
          + out_of_range_penalty * (length > max_len or length < min_len)

  Args:
    start: int, length normalization start offset.
    alpha: float, [0, 1.0],  length normalization power.
    min_len: int, minimum decode length.
    max_len: int, maximum decode lengths.
    out_of_range_penalty: float, penalty for lengths outside min len and max
      len. Use a negative number that penalize out of range decodes, does hard
      constraint if set to -inf.

  Returns:
    fn(log_probs_BxM, length)->scores_BxM: a function to normalize sum log
    probabilities of sequence with current decoding lengths.
  """

  def length_norm_fn(log_probs_BxM, length_int):
    """Normalize sum log probabilities given a sequence length."""
    dtype = log_probs_BxM.dtype
    norm_flt = torch.pow(torch.tensor((start + length_int) / (1. + start)),
                      alpha).cuda()
    log_probs_BxM /= norm_flt
    too_short_bool = torch.lt(torch.tensor(length_int), min_len).cuda()
    too_long_bool = torch.logical_and(torch.gt(torch.tensor(length_int), max_len), torch.tensor(max_len) > 0).cuda()
    out_of_range_bool = torch.logical_or(too_long_bool, too_short_bool).cuda()
    log_probs_BxM += out_of_range_penalty * out_of_range_bool.type(dtype)
    return log_probs_BxM

  return length_norm_fn


def map_structure(fn, obj):
    r"""Map a function over all elements in a (possibly nested) collection.
    
    Credit to https://github.com/asyml/texar-pytorch/blob/master/texar/torch/utils/utils.py 

    Args:
        fn (callable): The function to call on elements.
        obj: The collection to map function over.

    Returns:
        The collection in the same structure, with elements mapped.
    """

    if isinstance(obj, list):
        return [map_structure(fn, x) for x in obj]
    if isinstance(obj, tuple):
        if isinstance(obj, torch.Size):
            return fn(obj)
        # if hasattr(obj, '_fields'):  # namedtuple
        #     return type(obj)(*[map_structure(fn, x) for x in obj])
        else:
            return tuple(map_structure(fn, x) for x in obj)
    if isinstance(obj, dict):
        return {k: map_structure(fn, v) for k, v in obj.items()}
    if isinstance(obj, set):
        return {map_structure(fn, x) for x in obj}
    return fn(obj) 


def beam_search(symbols_to_logits_fn,
                init_seq_BxT,
                initial_cache_BxU,
                vocab_size,
                beam_size,
                length_norm_fn,
                eos_id=1):
  """Beam search.

  Args:
    symbols_to_logits_fn: fn(seq_BxT, cache_BxU, i) -> (logits_BxV, cache_BxU)
    init_seq_BxT: initial sequence ids.
    initial_cache_BxU: dictionary of tensors with shape BxU.
    vocab_size: vocabulary size.
    beam_size: beam size.
    length_norm_fn: length normalization function.
    eos_id: end of sequence.

  Returns:
    Tuple of (beams_BxMxT, scores_BxM). Beam searched sequences and scores.
  """
  shapes = list(init_seq_BxT.size())
  B, T = shapes[0], shapes[1]
  M, V = beam_size, vocab_size
  dtype = torch.float32
  int_dtype = init_seq_BxT.dtype

  # initialize.
  init_i = 0
  init_alive_seq_BxMxT = _expand_to_beam_size(init_seq_BxT.clone().cuda(), M)
  log_probs_1xM = torch.tensor([[0.] + [torch.finfo(dtype).min] * (M - 1)], dtype=dtype).cuda()
  init_alive_log_probs_BxM = log_probs_1xM.repeat((B, 1))
  init_alive_cache_BxMxU = map_structure(
      lambda t: _expand_to_beam_size(t, M), initial_cache_BxU)
  init_finished_seq_BxMxT = torch.zeros(tuple(init_alive_seq_BxMxT.size()), dtype=int_dtype).cuda()
  init_finished_scores_BxM = torch.zeros([B, M], dtype=dtype).cuda() + torch.finfo(dtype).min
  finished_alive_score = []
  final_finished_score = []

  # run loop.   
  with torch.no_grad():
    while init_i!=T:
      # Decode one step with beam
      logits_BMxV, cache_BMxU = symbols_to_logits_fn(
          _flatten_beam_dim(init_alive_seq_BxMxT),
          map_structure(_flatten_beam_dim, init_alive_cache_BxMxU), init_i)
      logits_BxMxV = _unflatten_beam_dim(logits_BMxV, M)
      new_cache_BxMxU = map_structure(lambda t: _unflatten_beam_dim(t, M),
                                              cache_BMxU)

      # select top 2 * beam_size and fill alive and finished.
      log_probs_BxMxV = logits_BxMxV - torch.logsumexp(
          logits_BxMxV, dim=2, keepdim=True)
      log_probs_BxMxV = log_probs_BxMxV.squeeze(dim=2)
      log_probs_BxMxV += torch.unsqueeze(init_alive_log_probs_BxM, dim=2)
      log_probs_BxMV = log_probs_BxMxV.view(B, -1)
      new_log_probs_Bx2M, topk_indices_Bx2M = torch.topk(log_probs_BxMV, k=2 * M)
      topk_beam_Bx2M = topk_indices_Bx2M // V
      topk_seq_Bx2MxT, new_cache_Bx2MxU = _gather_nested(
          [init_alive_seq_BxMxT, new_cache_BxMxU], topk_beam_Bx2M)
      topk_ids_Bx2M = topk_indices_Bx2M % V
      new_seq_Bx2MxT = _update_i(topk_seq_Bx2MxT, topk_ids_Bx2M, init_i).cuda()
      new_finished_flags_Bx2M = torch.any(torch.eq(new_seq_Bx2MxT, eos_id), dim=-1).type(dtype).cuda()

      # get new alive
      _, topk_alive_indices_BxM = torch.topk(
          new_log_probs_Bx2M + new_finished_flags_Bx2M * torch.finfo(dtype).min, k=M)
      (init_alive_seq_BxMxT, init_alive_log_probs_BxM, init_alive_cache_BxMxU) = _gather_nested(
          [new_seq_Bx2MxT, new_log_probs_Bx2M, new_cache_Bx2MxU],
          topk_alive_indices_BxM)

      # get new finished
      new_scores_Bx2M = length_norm_fn(new_log_probs_Bx2M, init_i + 1).cuda()
      new_scores_Bx2M += (1 - new_finished_flags_Bx2M) * torch.finfo(dtype).min
      finished_seq_Bx3MxT = torch.cat([init_finished_seq_BxMxT, new_seq_Bx2MxT],
                                      dim=1)
      finished_scores_Bx3M = torch.cat([init_finished_scores_BxM, new_scores_Bx2M],
                                      dim=1)
      _, topk_finished_indices_BxM = torch.topk(finished_scores_Bx3M, k=M)
      (init_finished_seq_BxMxT, init_finished_scores_BxM) = _gather_nested(
          [finished_seq_Bx3MxT, finished_scores_Bx3M], topk_finished_indices_BxM)
      
      finished_alive_score.append(init_alive_log_probs_BxM.unsqueeze(dim=2))
      final_finished_score.append(init_finished_scores_BxM.unsqueeze(dim=2))
      init_i += 1

  final_alive_seq_BxMxT = init_alive_seq_BxMxT.clone()
  final_alive_scores_BxM = init_alive_log_probs_BxM.clone()
  final_finished_seq_BxMxT = init_finished_seq_BxMxT.clone()
  final_finished_scores_BxM = init_finished_scores_BxM.clone()

  finished_alive_score = torch.stack(finished_alive_score).permute(1, 2, 0, 3)[:, :, :, -1]
  final_finished_score = torch.stack(final_finished_score).permute(1, 2, 0, 3)[:, :, :, -1]

  # process finished.
  final_finished_flag_BxMx1 = torch.any(
      torch.eq(final_finished_seq_BxMxT, eos_id), dim=-1, keepdim=True)
  final_seq_BxMxT = torch.where(
      final_finished_flag_BxMx1.repeat((1, 1, T)), final_finished_seq_BxMxT,
      final_alive_seq_BxMxT)
  final_scores_BxM = torch.where(
      torch.squeeze(final_finished_flag_BxMx1, dim=-1), final_finished_scores_BxM,
      final_alive_scores_BxM)
  final_scores_BxMxT = torch.where(
    final_finished_flag_BxMx1.repeat((1, 1, T)), final_finished_score,
    finished_alive_score)
  return final_seq_BxMxT, final_scores_BxMxT


def _update_i(tensor_BxNxT, updates_BxN, i):
  shapes = list(tensor_BxNxT.size())
  B, N, T = shapes[0], shapes[1], shapes[2]
  tensor_BNxT = tensor_BxNxT.view(-1, T).type(torch.LongTensor)
  updates_BN = updates_BxN.view(-1).type(torch.LongTensor)
  ind_BNx2 = torch.stack([
    torch.arange(B * N, dtype=torch.int64), 
    torch.full((B * N,), torch.tensor(i, dtype=torch.int64))], 
                         dim=-1).transpose(0, 1).type(torch.LongTensor)
  # tensor update sparse: tf.tensor_scatter_nd_update
  tensor_BNxT = tensor_BNxT.index_put_(list(tuple(ind_BNx2)), 
                               updates_BN, 
                               accumulate=True)
  return tensor_BNxT.view(B, N, T)


def _expand_to_beam_size(tensor_BxU, beam_size):
  tensor_Bx1xU = torch.unsqueeze(tensor_BxU, dim=1)
  tile_dims = [1] * tensor_Bx1xU.dim()
  tile_dims[1] = beam_size
  tensor_BxMxU = tensor_Bx1xU.repeat(tuple(tile_dims))
  return tensor_BxMxU


def _flatten_beam_dim(tensor_BxMxU):
  shapes = list(tensor_BxMxU.size())
  tensor_BMxU = tensor_BxMxU.view([shapes[0] * shapes[1]] + shapes[2:])
  return tensor_BMxU


def _unflatten_beam_dim(tensor_BMxU, M):
  shapes = list(tensor_BMxU.size())
  tensor_BxMxU = tensor_BMxU.view([shapes[0] // M, M] + shapes[1:])
  return tensor_BxMxU


def _gather_nested(nested_BxMxU, indices_BxN):

  def _gather_beam(tensor_BxMxU):
    result = []
    for p,i in zip(tensor_BxMxU, indices_BxN):
      r = p[i]
      result.append(r)
    tensor_BxNxU = torch.stack(result)
    # tf.gather(tensor_BxMxU, indices_BxN, batch_dims=1, axis=1)
    return tensor_BxNxU

  return map_structure(_gather_beam, nested_BxMxU)

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

"""Library for generative model decoding."""
# 
# pylint: disable=invalid-name

import torch
import torch.nn.functional as F
import modeling.beam_search as beam_search

EOS_ID = 1

def process_logits(logits_BxN, top_k=0, top_p=0.0, temperature=0.0):
  """Process logits using gumbel noise and mask top_k or top_p.

  The downstream task can perform probability sampling using gumbel-max trick
  (taking the argmax of processed logits) (Statistical theory of extreme values
  and some practical applications: a series of lectures. 1954).
  Use cases:
    greedy: top_k=0, top_p=0.0, temperature=0.0
    random sampling: top_k=0, top_p=0.0, temperature=1.0
    topk sampling: top_k=k, top_p=0.0, temperature=1.0
    nucleus sampling: top_k=0, top_p=p, temperature=1.0
    random sampling biased toward greedy: top_k=0, top_p=0.0, temperature=0.5
  Notations:
    B: batch_size, N: number of logits, K: topk value.
  Args:
    logits_BxN: tensor of [batch_size vocab_size]
    top_k: k in top_k sampling.
    top_p: probability in necleus sampling.
    temperature: gumbel noise sampling temperature.

  Returns:
    logits: processed logits which is original logits add gumbel noise and
    values outside top_k and top_p set to -inf.
  """
  if top_k > 0 and top_p > 0:
    raise ValueError(
        "Only one of the top_k and nucleus sampling should be specified.")
  
  shapes = list(logits_BxN.size())
  B, V = shapes[0], shapes[1]

  if top_k > 0:
    top_values_BxK, _ = torch.topk(logits_BxN, k=top_k, sorted=False)
    min_value_Bx1 = torch.min(top_values_BxK, dim=-1, keepdim=True)
    mask_BxN = torch.lt(logits_BxN, min_value_Bx1).type(logits_BxN.dtype)
    logits_BxN -= mask_BxN * logits_BxN.dtype.max

  if top_p > 0:
    sort_indices_BxN = torch.argsort(logits_BxN, dim=-1, descending=True)
    probs_BxN = torch.gather(F.softmax(logits_BxN), 
                             1, \
                            sort_indices_BxN[..., None].expand( \
                              *sort_indices_BxN.shape, \
                              F.softmax(logits_BxN).shape[-1])) 
    cumprobs_BxN = torch.cumsum(probs_BxN, dim=-1).roll(1, 0)
    cumprobs_BxN[0] = 0
    # The top 1 candidate always will not be masked.
    # This way ensures at least 1 indices will be selected.
    sort_mask_BxN = torch.gt(cumprobs_BxN, top_p).type(logits_BxN.dtype)
    batch_indices_BxN = torch.unsqueeze(torch.arange(B), dim=-1).repeat((1, V))
    # tensor update zeros: tf.scatter_nd
    torch_tensor = torch.zeros(shapes, sort_mask_BxN.dtype)
    torch_indices = torch.stack([batch_indices_BxN, sort_indices_BxN], dim=-1).transpose(0, 1)
    top_p_mask_BxN = torch_tensor.index_put_(list(tuple(torch_indices)), 
                            sort_mask_BxN, 
                            accumulate=True)
    logits_BxN -= top_p_mask_BxN * logits_BxN.dtype.max

  if temperature > 0:
    logits_shape = shapes
    uniform_noise_BxN = torch.rand(logits_shape)
    logits_BxN += -torch.log(-torch.log(uniform_noise_BxN)) * temperature
  return logits_BxN


def inplace_update_i(tensor_BxL, updates_B, i):
  """Inplace update a tensor. B: batch_size, L: tensor length."""
  torch
  B = list(tensor_BxL.size())
  batch_size = B[0]
  indices_Bx2 = torch.stack([
      torch.arange(batch_size, dtype=torch.int64),
      torch.full((batch_size,), torch.tensor(i))], dim=-1).transpose(0, 1)
  # tensor update sparse: tf.tensor_scatter_nd_update
  return tensor_BxL.index_put_(list(tuple(indices_Bx2)), 
                               updates_B, 
                               accumulate=True)


def left2right_decode(symbols_to_logits_fn,
                      context_BxU_dict,
                      batch_size,
                      max_decode_len,
                      vocab_size,
                      beam_size=1,
                      beam_start=5,
                      beam_alpha=0.6,
                      beam_min=0,
                      beam_max=-1,
                      temperature=0.0,
                      top_k=0,
                      top_p=0.0,
                      eos_id=EOS_ID):
  """left to right decode.

  Notations:
    B: batch_size, V: vocab_size, T: decode_len, U: undefined dimensions

  Args:
    symbols_to_logits_fn: logits = fn(decodes, context, i). Shoud take
      [batch_size, decoded_ids] and return [batch_size, vocab_size].
    context_BxU_dict: dict of Tensors.
    batch_size: int, decode batch size.
    max_decode_len: int, maximum number of steps to decode.
    vocab_size: int, output vocab size.
    beam_size: Number of beams to decode.
    beam_start: start length for scaling, default to 5.
    beam_alpha: Length penalty for decoding. Should be between 0 (shorter) and 1
      (longer), default to 0.6.
    beam_min: Minimum beam search lengths.
    beam_max: Maximum beam search lengths. Set -1 to use unlimited.
    temperature: Sampling temp for next token (0 for argmax), default to 0.0.
    top_k: Number of top symbols to consider at each time step, default to 0
      (consider all symbols).
    top_p: Nucleus sampling probability.
    eos_id: end of token id, default to 1.

  Returns:
    decodes: Tensor[batch, decode_len]
  """
  dtype = torch.int64
  # When beam_size=1, beam_search does not behave exactly like greedy.
  # This is due to using 2 * beam_size in grow_topk, and keep the top beam_size
  # ones that haven't reached EOS into alive.
  # In this case, alpha value for length penalty will take effect.
  if beam_size == 1:
    i = 0
    decodes_BxT = torch.zeros([batch_size, max_decode_len], dtype=dtype).cuda()
    cache_BxU_dict = context_BxU_dict
    logit_ls = []
    # torch.zeros([max_decode_len], dtype=torch.float32).cuda()
    logits_Bx1xV = torch.zeros([batch_size, 1, vocab_size], dtype=torch.float32).cuda() 
    finished_B = torch.any(torch.eq(decodes_BxT, EOS_ID), dim=1).cuda()

    while torch.logical_and(torch.tensor(i).cuda() < torch.tensor(max_decode_len).cuda(),
                            torch.logical_not(torch.all(finished_B))):
      
      logits_Bx1xV = symbols_to_logits_fn(decodes_BxT, cache_BxU_dict, i)
      logits_BxV = torch.squeeze(logits_Bx1xV, dim=1)
      logit_ls.append(logits_Bx1xV)
      logits_BxV = process_logits(logits_BxV, top_k, top_p, temperature)
      decodes_BxT = inplace_update_i(decodes_BxT, torch.argmax(logits_BxV, -1), i)
      finished_B = torch.any(torch.eq(decodes_BxT, EOS_ID), dim=1).cuda()
      i += 1
      
    decodes = decodes_BxT.clone()
    logits = logits_Bx1xV
    
    logit_ls = torch.stack(logit_ls)
    logit_st = logit_ls.permute(1, 0, 3, 2)
    del(logit_ls)
    logit_st = logit_st[:,:,:,-1]
    scores = F.softmax(logit_st, dim=-1).cuda()
    del(logit_st)
    return decodes, scores

  else:
    def symbols_to_logits_fn_with_sampling(decodes_BxT, states_BxU_dict, i):
      logits_BxV = symbols_to_logits_fn(decodes_BxT, states_BxU_dict, i)
      logits_BxV = process_logits(logits_BxV, top_k, top_p, temperature)
      return logits_BxV, states_BxU_dict

    length_norm_fn = beam_search.length_normalization(beam_start, beam_alpha,
                                                      beam_min, beam_max, -1e3)
    beams, scores = beam_search.beam_search(
        symbols_to_logits_fn_with_sampling,
        torch.zeros([batch_size, max_decode_len], dtype=torch.int32).cuda(),
        context_BxU_dict, vocab_size, beam_size, length_norm_fn, eos_id)
    return beams[:, 0, :].type(dtype), scores # log_prob_score

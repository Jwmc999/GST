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

"""Attention layers.

Notations:
  B: batch_size, I: max_input_len, M: max_memory_len, D: hidden_size,
  H: number of heads, Dh: hidden_size per head, Di: input dimension.
"""
# 
# pylint: disable=invalid-name

import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F

def split_heads(tensor_BxIxD, num_heads):
  input_shape = list(tensor_BxIxD.size())
  B, I, D = input_shape[0], input_shape[1], input_shape[2]
  tensor_BxIxHxD = tensor_BxIxD.view(B, I, num_heads, D // num_heads)
  tensor_BxHxIxD = tensor_BxIxHxD.permute(0, 2, 1, 3)
  return tensor_BxHxIxD


class Attention(Module):
  """Multihead scaled dot product attention."""

  def __init__(self, hidden_size, num_heads, attention_dropout):
    super(Attention, self).__init__()

    if hidden_size % num_heads != 0:
      raise ValueError("Number of attention heads must divide hidden size")

    self._q_layer = nn.Linear(hidden_size, hidden_size, bias=False).cuda()
    self._k_layer = nn.Linear(hidden_size, hidden_size, bias=False).cuda()
    self._v_layer = nn.Linear(hidden_size, hidden_size, bias=False).cuda()
    self._output_layer = nn.Linear(hidden_size, hidden_size, bias=False).cuda()
    self._num_heads = num_heads
    self._hidden_size = hidden_size
    self._attention_dropout = attention_dropout

  def forward(self,
               input_BxIxDi,
               memory_BxMxDi,
               bias_BxIxM,
               training,
               cache=None,
               decode_i=None):

    input_shape = list(input_BxIxDi.size())
    B, I = input_shape[0], input_shape[1]
    M, H, D = memory_BxMxDi.shape[1], self._num_heads, self._hidden_size
    dtype = memory_BxMxDi.dtype

    q_BxHxIxDh = split_heads(self._q_layer(input_BxIxDi), H)
    q_BxHxIxDh *= (D // H)**-0.5
    k_BxHxMxDh = split_heads(self._k_layer(memory_BxMxDi), H)
    v_BxHxMxDh = split_heads(self._v_layer(memory_BxMxDi), H)

    # cache saves previous activations before time decode_i during TPU decoding.
    if cache is not None and decode_i is not None:
      M = cache["k"].shape[2]
      indices_1x1xMx1 = F.one_hot(decode_i, M).view(1, 1, M, 1)
      k_BxHxMxDh = cache["k"].cuda() + k_BxHxMxDh * indices_1x1xMx1
      v_BxHxMxDh = cache["v"].cuda() + v_BxHxMxDh * indices_1x1xMx1
      cache["k"] = k_BxHxMxDh
      cache["v"] = v_BxHxMxDh
    bias_BxHxIxM = torch.unsqueeze(bias_BxIxM, 1)
    logits_BxHxIxM = torch.matmul(
        q_BxHxIxDh, k_BxHxMxDh.permute(0, 1, 3, 2)) + bias_BxHxIxM
    alignment_BxHxIxM = F.softmax(logits_BxHxIxM)
    if training:
        dropout = nn.Dropout(p=self._attention_dropout)
        alignment_BxHxIxM = dropout(alignment_BxHxIxM) 
    outputs_BxHxIxDh = torch.matmul(alignment_BxHxIxM, v_BxHxMxDh)
    outputs_BxIxD = outputs_BxHxIxDh.permute(0, 2, 1, 3).reshape(B, I, D)
    outputs_BxIxD = self._output_layer(outputs_BxIxD)
    return outputs_BxIxD


class SelfAttention(Attention):
  """Multihead scaled dot product self-attention."""
  def __init__(self, hidden_size, num_heads, attention_dropout):
    super(SelfAttention, self).__init__(hidden_size, num_heads, attention_dropout)
    self.attention = Attention(hidden_size, num_heads, attention_dropout)

  def forward(self, x, bias, training, cache=None, decode_i=None):
    return self.attention(
        x, x, bias, training, cache=cache, decode_i=decode_i)


def ids_to_bias(ids_BxI, dtype=torch.float32, padding_id=0):
  """Convert ids to attention bias for attention."""
  pad_BxI = torch.equal(ids_BxI, padding_id).type(dtype)
  bias_Bx1xI = torch.unsqueeze(pad_BxI * torch.finfo(dtype).min, 1)
  return bias_Bx1xI


def upper_triangle_bias(D, dtype=torch.float32):
  """Create a upper triangle matrix for decoding bias."""
  upper_triangle_DxD = 1 - torch.tril(torch.ones([D, D], dtype=dtype)) 
  tensor_1xDxD = torch.unsqueeze(upper_triangle_DxD * torch.finfo(dtype).min, 0)
  return tensor_1xDxD

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

"""Timing layers.

Notations:
B: batch_size, I: input_length, D: hidden_size, N: num_timescales
"""
# 
# pylint: disable=invalid-name

import math
import torch

_MIN_TIMESCALE = 1.0
_MAX_TIMESCALE = 1.0e4


def add_time_signal(inputs_BxIxD, start_index=None):
  """Adds a transformer-style timing signal to inputs.

  Using periodic signals as in https://arxiv.org/abs/1706.03762.
  Generalized to allow each example in a batch to begin at a different index.

  Args:
    inputs_BxIxD: input representation.
    start_index: tensor of starting pos. [batch_size]

  Returns:
    output: representation with time signal added, same shape as input.
  """

  dtype = inputs_BxIxD.dtype
  states_shape = list(inputs_BxIxD.size())
  B = states_shape[0]
  I= states_shape[1]
  D = states_shape[2]

  if D % 2 != 0:
    raise ValueError("Input dimension must be even.")
  if start_index is not None:
    start_Bx1 = start_index
  else:
    start_Bx1 = torch.zeros([B, 1], dtype=torch.int32).cuda()

  pos_1xI = torch.unsqueeze(torch.arange(I), 0).cuda()
  pos_BxI = torch.tile(pos_1xI, (B, 1)) + start_Bx1
  pos_BxI = pos_BxI.type(dtype)
  N = D // 2
  log_time_incr = (
      math.log(_MAX_TIMESCALE / _MIN_TIMESCALE) /
      torch.maximum(torch.tensor(N - 1).type(dtype), torch.tensor(1))).cuda()
  inv_scale_N = _MIN_TIMESCALE * torch.exp(
      torch.arange(N, dtype=dtype).cuda() * -log_time_incr)
  time_BxIxN = torch.unsqueeze(pos_BxI, 2) * inv_scale_N.view(1, 1, -1)
  signal_BxIxD = torch.cat((torch.sin(time_BxIxN), torch.cos(time_BxIxN)), dim=2)
  signal_BxIxD = signal_BxIxD.view(B, I, D)
  return inputs_BxIxD + signal_BxIxD

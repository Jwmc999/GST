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

import six
import tensorflow as tf


def split_heads(tensor_BxIxD, num_heads):
  input_shape = get_shape_list(tensor_BxIxD)
  B, I, D = input_shape[0], input_shape[1], input_shape[2]
  tensor_BxIxHxD = tf.reshape(tensor_BxIxD, [B, I, num_heads, D // num_heads])
  tensor_BxHxIxD = tf.transpose(tensor_BxIxHxD, [0, 2, 1, 3])
  return tensor_BxHxIxD


class Attention(tf.keras.layers.Layer):
  """Multihead scaled dot product attention."""

  def __init__(self, hidden_size, num_heads, attention_dropout):
    super(Attention, self).__init__()

    if hidden_size % num_heads != 0:
      raise ValueError("Number of attention heads must divide hidden size")

    self._q_layer = tf.keras.layers.Dense(hidden_size, use_bias=False, name="q_proj")
    self._k_layer = tf.keras.layers.Dense(hidden_size, use_bias=False, name="k_proj")
    self._v_layer = tf.keras.layers.Dense(hidden_size, use_bias=False, name="v_proj")
    self._output_layer = tf.keras.layers.Dense(
        hidden_size, use_bias=False, name="output_proj")
    self._num_heads = num_heads
    self._hidden_size = hidden_size
    self._attention_dropout = attention_dropout

  def __call__(self,
               input_BxIxDi,
               memory_BxMxDi,
               bias_BxIxM,
               training,
               cache=None,
               decode_i=None):

    input_shape = get_shape_list(input_BxIxDi, expected_rank=[2,3])
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
      indices_1x1xMx1 = tf.reshape(
          tf.one_hot(decode_i, M, dtype=dtype), [1, 1, M, 1])
      k_BxHxMxDh = cache["k"] + k_BxHxMxDh * indices_1x1xMx1
      v_BxHxMxDh = cache["v"] + v_BxHxMxDh * indices_1x1xMx1
      cache["k"] = k_BxHxMxDh
      cache["v"] = v_BxHxMxDh
    bias_BxHxIxM = tf.expand_dims(bias_BxIxM, axis=1)
    logits_BxHxIxM = tf.matmul(
        q_BxHxIxDh, k_BxHxMxDh, transpose_b=True) + bias_BxHxIxM
    alignment_BxHxIxM = tf.nn.softmax(logits_BxHxIxM)
    if training:
      alignment_BxHxIxM = tf.compat.v2.nn.dropout(
          alignment_BxHxIxM, self._attention_dropout, noise_shape=[1, 1, I, M])
    outputs_BxHxIxDh = tf.matmul(alignment_BxHxIxM, v_BxHxMxDh)
    outputs_BxIxD = tf.reshape(
        tf.transpose(outputs_BxHxIxDh, [0, 2, 1, 3]), [B, I, D])
    outputs_BxIxD = self._output_layer(outputs_BxIxD)
    return outputs_BxIxD, alignment_BxHxIxM


class SelfAttention(Attention):
  """Multihead scaled dot product self-attention."""

  def __call__(self, x, bias, training, cache=None, decode_i=None):
    return super(SelfAttention, self).__call__(
        x, x, bias, training, cache=cache, decode_i=decode_i)


def ids_to_bias(ids_BxI, dtype=tf.float32, padding_id=0):
  """Convert ids to attention bias for attention."""
  pad_BxI = tf.cast(tf.equal(ids_BxI, padding_id), dtype)
  bias_Bx1xI = tf.expand_dims(pad_BxI * tf.float32.min, axis=1)
  return bias_Bx1xI

def upper_triangle_bias(D, dtype=tf.float32):
  """Create a upper triangle matrix for decoding bias."""
  upper_triangle_DxD = 1 - tf.compat.v1.matrix_band_part(
      tf.ones([D, D], dtype=dtype), -1, 0)
  tensor_1xDxD = tf.expand_dims(upper_triangle_DxD * tf.float32.min, axis=0)
  return tensor_1xDxD


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    # if name is None:
    #     name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]

    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    # if name is None:
    #     name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.compat.v1.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape),
             str(expected_rank)))
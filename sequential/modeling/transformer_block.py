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

"""Transformer block.

From "Attention Is All You Need", https://arxiv.org/abs/1706.03762.

Notations:
  B: batch_size, I: max_input_len, M: max_memory_len, D: hidden_size
"""
# 
# pylint: disable=invalid-name
# pylint: disable=g-long-lambda
import six
import modeling.attention as attention
import tensorflow as tf

class TransformerBlock(object):
  """Transformer block.

  Attention block of self-attention, attention over external memory, and
  feedforward network.
  Initialize the block with
    block = TransformerBlock(hidden_size, num_heads, dropout)
  To create an encoder self attention layer, use
    x = block(x, x_bias, None, None)
  To create a decoder attention layer, use
    y = block(y, upper_triangle_bias, x, x_bias)
  """
  def __init__(self, hidden_size, hidden_act, intermediate_size, num_heads, dropout):
    # super(TransformerBlock, self).__init__()
    self._self_attn_layer = attention.SelfAttention(hidden_size, num_heads,
                                                    dropout)
    self._attn_layer = attention.Attention(hidden_size, num_heads, dropout)
    self._activation_layer = tf.keras.layers.Dense(intermediate_size, 
                                             activation=get_activation(hidden_act)) 
    self._output_layer = tf.keras.layers.Dense(hidden_size)
    self._dropout_prob = dropout

  def __call__(self,
               is_training,
               inputs_BxIxD,
               bias_BxIxI,
               memory_BxMxD,
               bias_BxIxM,
               cache=None,
               decode_i=None):
    s_BxIxD = inputs_BxIxD
    with tf.compat.v1.variable_scope("self_attention"):
      y_BxIxD = layer_norm(s_BxIxD)
      y_BxIxD, self_score = self._self_attn_layer(
          y_BxIxD, bias_BxIxI, is_training, cache=cache, decode_i=decode_i)
      s_BxIxD += dropout(y_BxIxD, self._dropout_prob)
    if memory_BxMxD is not None:
      with tf.compat.v1.variable_scope("masked_attention"):
        y_BxIxD = layer_norm(s_BxIxD)
        y_BxIxD, mask_score = self._attn_layer(y_BxIxD, memory_BxMxD, bias_BxIxM, is_training)
        s_BxIxD += dropout(y_BxIxD, self._dropout_prob)
    with tf.compat.v1.variable_scope("ffn"):
      y_BxIxD = layer_norm(s_BxIxD)
      y_BxIxD = dropout(self._activation_layer(y_BxIxD), self._dropout_prob)
      s_BxIxD += dropout(self._output_layer(y_BxIxD), self._dropout_prob)
    return s_BxIxD, self_score

def stack(layers,
          is_training,
          inputs_BxIxD,
          bias_BxIxI, # attention mask
          memory_BxMxD,
          bias_BxIxM,
          cache=None,
          decode_i=None):
  """Stack AttentionBlock layers."""
  if (memory_BxMxD is None) != (bias_BxIxM is None):
    raise ValueError("memory and memory_bias need to be provided together.")
  s_BxIxD = inputs_BxIxD
  for i, layer in enumerate(layers):
    with tf.compat.v1.variable_scope("layer_%d" % i):
      s_BxIxD, score = layer(
          is_training,
          s_BxIxD,
          bias_BxIxI,
          memory_BxMxD,
          bias_BxIxM,
          cache=cache[str(i)] if cache is not None else None,
          decode_i=decode_i)
  return s_BxIxD, score

def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    layer_norm_cls = tf.keras.layers.LayerNormalization(axis=-1, name=name)
    return layer_norm_cls(input_tensor)

def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
        input_tensor: float Tensor.
        dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
        A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output

def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.compat.v1.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf

def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)
    
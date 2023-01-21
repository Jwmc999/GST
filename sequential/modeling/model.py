"""The main Pegasus model and related functions."""

import collections
import copy
import json
import re
import six
import tensorflow as tf
import modeling.attention as attention
import modeling.transformer_block as transformer_block
import utils.timing as timing
import modeling.embedding as embedding
import modeling.decoding as decoding


class PegasusConfig(object):
    """Configuration for `PegasusModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_encoder_layers=12,
                 num_decoder_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 label_smoothing=0.1):
        """Constructs PegasusConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `PegasusModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_encoder_layers: Number of hidden layers in the Transformer encoder.
            num_decoder_layers: Number of hidden layers in the Transformer decoder.
                If None, use `num_encoder_layers`
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder and decoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder and decoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder, decoder, and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, decoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `PegasusModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers # encoder_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_encoder_layers # decoder_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `PegasusConfig` from a Python dictionary of parameters."""
        config = PegasusConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `PegasusConfig` from a json file of parameters."""
        with tf.compat.v1.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class GST(tf.keras.layers.Layer):
    """Standard Transformer models.

    Models contain embedding, encoding, and loss functions, and expect text ids as
    inputs. All models have same format as below:
    
    ```python
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
    
    config = PegasusConfig(vocab_size=32000, hidden_size=512,
        num_encoder_layers=8, num_decoder_layers=8, 
        num_attention_heads=6, intermediate_size=1024)

    model = PegasusModel(config=config, training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    loss, logits = model(training)
    Features and outputs are dictionary of tensors. Features usually inlucdes inputs
    and targets ids.
    """
    def __init__(self,
                 config,
                 use_one_hot_embeddings=True,
                 scope=None):
        """Constructor for PegasusModel.

        Args:
            config: `PegasusConfig` instance.
            training: bool. True for training model, false for eval model. Controls
                whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
                it is must faster if this is True, on the CPU or GPU, it is faster if
                this is False.
            scope: (optional) variable scope. Defaults to "mlm".

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        config = copy.deepcopy(config)
        super(GST, self).__init__()
        
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_attention_heads = config.num_attention_heads
        self.label_smoothing = config.label_smoothing
        self.hidden_act = config.hidden_act
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.initializer_range = config.initializer_range
        self.type_vocab_size = config.type_vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_position_embeddings = config.max_position_embeddings
        self.scope = scope
        
        self._dtype = tf.float32
        self._embedding_layer = embedding.Embedding(self.vocab_size, 
                                                    self.hidden_size,
                                                    "weights", 
                                                    self._dtype)
        block_fn = lambda: transformer_block.TransformerBlock(
            hidden_size=config.hidden_size, 
            hidden_act=config.hidden_act, 
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads, 
            dropout=config.hidden_dropout_prob)
        self._encoder_layers = [block_fn() for _ in range(config.num_encoder_layers)]
        self._decoder_layers = [block_fn() for _ in range(config.num_decoder_layers)]
        self._dropout_fn = lambda x, training: tf.compat.v2.nn.dropout(
            x, config.hidden_dropout_prob, noise_shape=[tf.shape(x)[0], 1, tf.shape(x)[2]]) if training else x  
    
    def _decoder(self, inputs, training):
        """Create pretraining model. 

        Args:
            inputs: dictionary of tensors including "inputs" [batch, input_len] and
                "targets" [batch, output_len]

        Returns:
            Tuple of (loss, outputs): Loss is a scalar. Output is a dictionary of
             tensors, containing model's output logits.
        """
        global self_score
        labels = inputs
        targets_BxT = labels['input_ids'] 

        bias_1xTxT = attention.upper_triangle_bias(
            tf.shape(targets_BxT)[1], self._dtype)
        states_BxTxD = self._embedding_layer(targets_BxT, True)
        states_BxTxD = tf.pad(states_BxTxD, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        states_BxTxD = timing.add_time_signal(states_BxTxD)
        states_BxTxD = self._dropout_fn(states_BxTxD, training)
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE):
            states_BxTxD, self_score = transformer_block.stack(self._decoder_layers, training,
                                                    states_BxTxD, bias_1xTxT,
                                                    None, None)
            states_BxTxD = transformer_block.layer_norm(states_BxTxD)
        # embeding weights to logits
        logits_BxTxV = self._embedding_layer(states_BxTxD, False)
        targets_mask_BxT = tf.cast(tf.greater(targets_BxT, 0), self._dtype)
        loss = tf.compat.v1.losses.softmax_cross_entropy(
            tf.one_hot(targets_BxT, self.vocab_size),
            logits_BxTxV,
            label_smoothing=self.label_smoothing,
            weights=targets_mask_BxT)
        return loss, {"logits": logits_BxTxV}, self_score

    
    def _predict(self, labels, max_decode_len, beam_size, batch_size, **beam_kwargs):
        """Predict. Fine tuning model."""
        cache = collections.defaultdict(list)

        # Initialize cache for decoding
        B, D = batch_size, self.hidden_size
        T, V, H = max_decode_len, self.vocab_size, self.num_attention_heads

        bias_1xTxT = attention.upper_triangle_bias(T, self._dtype)
        for i in range(len(self._decoder_layers)):
            cache[str(i)] = {
                "k": tf.zeros([B, H, T, D // H], self._dtype),
                "v": tf.zeros([B, H, T, D // H], self._dtype)
            }
            
        def symbols_to_logits_fn(dec_BxT, context, i):
            """Decode loop."""
            global self_score
            dec_shape = attention.get_shape_list(dec_BxT)
            dec_Bx1 = tf.slice(dec_BxT, [0, tf.maximum(tf.cast(0, i.dtype), i - 1)],
                                [dec_shape[0], 1])
            bias_1x1xT = tf.slice(bias_1xTxT, [0, i, 0], [1, 1, T])
            dec_Bx1xD = self._embedding_layer(dec_Bx1, True)
            dec_Bx1xD *= tf.cast(tf.greater(i, 0), self._dtype)
            dec_Bx1xD = timing.add_time_signal(dec_Bx1xD, start_index=i)
            with tf.compat.v1.variable_scope("decoder"):    
                dec_Bx1xD, self_score = transformer_block.stack(self._decoder_layers, False,
                                                    dec_Bx1xD, bias_1x1xT,
                                                    None,
                                                    None, context, i)
                dec_Bx1xD = transformer_block.layer_norm(dec_Bx1xD)
            # embeding weights to logits
            logits_Bx1xV = self._embedding_layer(dec_Bx1xD, False)  
            return logits_Bx1xV

        decodes_BxT, scores = decoding.left2right_decode(symbols_to_logits_fn, cache, B, T,
                                                V, beam_size, **beam_kwargs)
        print("decode shape", decodes_BxT.shape)
        print("score shape", scores.shape)

        if beam_size >1:
          decodes_BxT = tf.reshape(decodes_BxT , [-1])
          scores = tf.math.exp(decodes_BxT) / (tf.math.exp(decodes_BxT)).sum()
          scores = tf.math.log(scores)
        return {"outputs": decodes_BxT}, {"gap_sg_log_probs": scores}, self_score

    def __call__(self, inputs, training, max_dec_len=None, beam_size=None, batch_size=None, **beam_kwargs):
        if training:
            predictions = self._decoder(inputs, training)
        else:
            predictions = self._predict(inputs, max_dec_len, beam_size, batch_size, **beam_kwargs)
        
        return predictions

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = dict()
    initialized_variable_names = dict()

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        # assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)





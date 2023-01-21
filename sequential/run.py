import os
import sys
import pickle
import json

import tensorflow as tf
import numpy as np
import random

from modeling.model import *
import utils.optimization as optimization


use_tpu = False
training= True
use_pop_random = True
init_checkpoint = None if training else "models/test/ml-1m/summarization/model.ckpt-10000"

pegi_config_file = 'data_dir/pegi_config_ml-1m_64.json'
train_input_file = 'data_dir/summarization/ml-1m.train.tfrecord'
test_input_file = 'data_dir/summarization/ml-1m.test.tfrecord'
vocab_filename = 'data_dir/summarization/ml-1m.vocab'
user_history_filename = 'data_dir/summarization/ml-1m.his'
checkpoint_dir = 'models/test/ml-1m'
save_checkpoints_steps = 1000
max_seq_length = 20
max_predictions_per_seq = 2
batch_size = 32
samples = -1
learning_rate = 1e-2

num_warmup_steps = 100
beam_size = 1
max_dec_len = max_predictions_per_seq
task = 'summarization'

num_train_steps = 10000


# Evaluation Hooks added as evaluation metric: hit ratio, NDCG
class EvalHooks(tf.compat.v1.train.SessionRunHook):
    def __init__(self):
        print('run init')

    def begin(self):
        self.valid_user = 0.0

        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.ap = 0.0
        self.result = []

        np.random.seed(12345) #12345
        # random.seed(82)

        self.vocab = None

        if user_history_filename is not None:
            print('load user history from :' + user_history_filename)
            with open(user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if vocab_filename is not None:
            print('load vocab from :' + vocab_filename)
            with open(vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            # ids: 'list'

            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            sum_value = np.sum([x for x in values])
            self.probability = [value / sum_value for value in values]

    def end(self, session):        
        ndcg_1, hit_1, ndcg_5, hit_5, ndcg_10, hit_10, ap, valid_user = self.result[0]

        evals = {"ndcg@1": ndcg_1 / valid_user,
                 "hit@1": hit_1 / valid_user,
                 "ndcg@5": ndcg_5 / valid_user,
                 "hit@5": hit_5 / valid_user,
                 "ndcg@10": ndcg_10 / valid_user,
                 "hit@10": hit_10 / valid_user,
                 "ap": ap / valid_user,
                 "valid_user": valid_user}

        ckpt_dir = checkpoint_dir + '/' + task
        pegi_config = PegasusConfig.from_json_file(pegi_config_file)
        output_eval_file = os.path.join(ckpt_dir,
                                        "score_results.txt")
        with open(output_eval_file, "w",  encoding='utf-8') as writer:
            print("***** Score results *****")
            print(pegi_config.to_json_string())
            writer.write(pegi_config.to_json_string() + '\n')
            for key in evals.keys():
                print("%s = %s" % (key, str(evals[key])))
                writer.write("%s = %s\n" % (key, str(evals[key])))

    def before_run(self, run_context):
        variables = tf.compat.v1.get_collection('eval_sp')
        return tf.compat.v1.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        gap_sg_log_probs, output_ids, gap_sg_ids, label_info = run_values.results
        gap_sg_log_probs = gap_sg_log_probs.reshape(
            (-1, max_predictions_per_seq, gap_sg_log_probs.shape[1]))

        for idx in range(len(output_ids)):
            rated = set(output_ids[idx])
            rated.add(0)
            rated.add(gap_sg_ids[idx][0])
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(label_info[idx][0])][0])
            item_idx = [gap_sg_ids[idx][0]]
            # here we need more consideration
            gap_sg_log_probs_elem = gap_sg_log_probs[idx, 0]
            size_of_prob = len(self.ids) + 1  # len(gap_sg_log_probs_elem)
            if use_pop_random:
                if self.vocab is not None:
                    while len(item_idx) < 101:
                        sampled_ids = np.random.choice(self.ids, 101, replace=False, p=self.probability)
                        sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                        item_idx.extend(sampled_ids[:])
                    item_idx = item_idx[:101]
            else:
                # print("evaluation random -> ")
                for _ in range(100):
                    t = np.random.randint(1, size_of_prob)
                    while t in rated:
                        t = np.random.randint(1, size_of_prob)
                    item_idx.append(t)

            predictions = -gap_sg_log_probs_elem[item_idx]
            rank = predictions.argsort().argsort()[0]

            self.valid_user += 1

            if self.valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            if rank < 1:
                self.ndcg_1 += 1
                self.hit_1 += 1
            if rank < 5:
                self.ndcg_5 += 1 / np.log2(rank + 2)
                self.hit_5 += 1
            if rank < 10:
                self.ndcg_10 += 1 / np.log2(rank + 2)
                self.hit_10 += 1

            self.ap += 1.0 / (rank + 1)

            if self.valid_user == 6040.0:
                self.result.append([self.ndcg_1, self.hit_1, \
                                    self.ndcg_5, self.hit_5, \
                                    self.ndcg_10, self.hit_10, \
                                    self.ap, self.valid_user])
                print('last user')
                break


 
def input_fn_builder(input_files,
                    max_seq_length,
                    max_predictions_per_seq,
                    batch_size,
                    training):
    def parser(file_input,
            training,
            max_seq_length,
            max_predictions_per_seq,
            batch_size,
            num_cpu_threads=4):
        """Parse TFRecord to BatchDataset, and then to Dict

        Args:
            file_input: `.tfrecord` file
            is_encoder: bool. If encoder, `name_to_features` with `input_ids`.
                If decoder, `name_to_features` with `output_ids`.
            training: bool. If training, shuffle dataset for parallel processsing.
                For eval, we don't need to shuffle dataset.
            max_seq_length: int. Must equal to `max_seq_length` of `gen_data.py` and `gen_sess.py`
            max_predictions_per_seq: int. Must equal to `max_predictions_per_seq` of `gen_data.py` and `gen_sess.py`
                If `max_predictions_per_seq` to large, can occur error.
            batch_size: int. batch size
            num_cpu_threads: int.

        Returns:
            dict of tf.Tensor

        example
        < Dict shapes:
                {info: (None, 1),
                input_ids: (None, 200),
                input_mask: (None, 200),
                masked_lm_ids: (None, 20),
                masked_lm_positions: (None, 20),
                masked_lm_weights: (None, 20)},
                types: {info: tf.int32,
                        input_ids: tf.int32,
                        input_mask: tf.int32,
                        masked_lm_ids: tf.int32,
                        masked_lm_positions: tf.int32,
                        masked_lm_weights: tf.float32} >
        """


        name_to_features = {
            "info":
                tf.io.FixedLenFeature([1], tf.int64),  # [user]
            "input_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64, default_value=[0] * max_seq_length),
            "input_mask":
                tf.io.FixedLenFeature([max_seq_length], tf.int64, default_value=[0] * max_seq_length),
            "masked_lm_positions":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64,
                                                        default_value=[0] * max_predictions_per_seq),
            "masked_lm_ids":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32,
                                                default_value=[0.0] * max_predictions_per_seq),
                            }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if training:
            d = tf.data.TFRecordDataset(file_input)
            d = d.repeat()
            d = d.shuffle(buffer_size=10)

        else:
            d = tf.data.TFRecordDataset(file_input)

        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)

        # tf.BatchDataset
        d = d.batch(batch_size=batch_size).take(samples)
        return d

    def input_fn():
        # Parse TFRecord to BatchDataset, and then dict of tf.Tensor
        # decoder
        dec = parser(file_input=input_files,
                    training=training,
                    max_seq_length=max_seq_length,
                    max_predictions_per_seq=max_predictions_per_seq,
                    batch_size=batch_size)
        return dec

    return input_fn


def get_output(_loss, log_probs, vocab_size, label_ids, label_weights, training):
    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])
    one_hot_labels = tf.one_hot(
        label_ids, depth=vocab_size, dtype=tf.float32)

    if training:
        _log_probs = tf.nn.softmax(log_probs["logits"], -1)
        per_example_loss = -tf.reduce_sum(
            input_tensor=_log_probs * one_hot_labels, axis=[-1])
        return (_loss, per_example_loss)
    else:
        _log_probs = log_probs["gap_sg_log_probs"]
        _log_probs = tf.reshape(
            _log_probs, [-1, _log_probs.shape[-1]])
        per_example_loss = -tf.reduce_sum(
            input_tensor=_log_probs * one_hot_labels, axis=[-1])
        # numerator = tf.reduce_sum(input_tensor=label_weights * per_example_loss)
        # denominator = tf.reduce_sum(input_tensor=label_weights) + 1e-5
        # loss = numerator / denominator
        return (tf.constant(0.0), per_example_loss) #(loss, per_example_loss)


def model_fn_builder(pegi_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""


    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        print("*** Features ***")
        for name in sorted(features.keys()):
            print("name = %s, shape = %s" % (name, tf.shape(features[name])))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = GST(
            config=pegi_config,
            use_one_hot_embeddings=use_one_hot_embeddings)

        inputs = features

        result, _log, self_score = model(inputs=inputs,
                             training=is_training,
                             max_dec_len=params["max_dec_len"],
                             beam_size=params["beam_size"],
                             batch_size=params['batch_size'])
        _tar = tf.one_hot(inputs["input_ids"], model.vocab_size)

        (total_loss, example_loss) = get_output(result,
                                _log,
                                pegi_config.vocab_size,
                                features["masked_lm_ids"],
                                features["masked_lm_weights"],
                                training=is_training)

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None

        if init_checkpoint:
            # Manually load the latest checkpoint
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("  name = %s, shape = %s%s" % (var.name, var.shape,
                            init_string))

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)
            
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(loss, log_probs, label_ids, pred_ids):
                loss = tf.compat.v1.metrics.mean(values=loss)
                log_probs = tf.compat.v1.metrics.mean(
                    values=log_probs,
                    weights=tf.cast(tf.not_equal(label_ids, 0), tf.float32))
                metric_dict = {
                    "prediction_loss": loss,
                    "log_likelihood": log_probs,
                }
                return metric_dict

            metrics_fn = metric_fn(example_loss, _log["gap_sg_log_probs"],
                                    features["masked_lm_ids"], features["masked_lm_ids"])

            gap_sg_log_probs = _log["gap_sg_log_probs"]
            gap_sg_log_probs = tf.reshape(
                gap_sg_log_probs, [-1, gap_sg_log_probs.shape[-1]])

            tf.compat.v1.add_to_collection('eval_sp', gap_sg_log_probs)
            tf.compat.v1.add_to_collection('eval_sp', features["input_ids"])
            tf.compat.v1.add_to_collection('eval_sp', features["masked_lm_ids"])
            tf.compat.v1.add_to_collection('eval_sp', features["info"])

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                predictions=_log, #_log["gap_sg_log_probs"]
                eval_metric_ops=metrics_fn,
                scaffold=scaffold_fn)
        else:
            print("Wrong Mode. Choose from TRAIN, EVAL.")

        return output_spec

    return model_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example.

    example: 'dict'. TFExample
    keys: ['info', 'input_ids', 'input_mask', 'masked_lm_ids', 'masked_lm_positions', 'masked_lm_weights']
    value of `info`: Tensor("Cast:0", shape=(1,), dtype=int32)
    `dict` to `list`
    {'info': <tf.Tensor 'Cast:0' shape=(1,) dtype=int32>,
    'input_ids': <tf.Tensor 'Cast_1:0' shape=(200,) dtype=int32>,
    'input_mask': <tf.Tensor 'Cast_2:0' shape=(200,) dtype=int32>,
    'masked_lm_ids': <tf.Tensor 'Cast_3:0' shape=(20,) dtype=int32>,
    'masked_lm_positions': <tf.Tensor 'Cast_4:0' shape=(20,) dtype=int32>,
    'masked_lm_weights': <tf.Tensor 'ParseSingleExample/ParseExample/ParseExampleV2:5' shape=(20,) dtype=float32>}
    """

    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def main(_):
    tf.random.set_seed(12345)

    pegi_config = PegasusConfig.from_json_file(pegi_config_file)

    ckpt_dir = checkpoint_dir + '/' + task
    tf.compat.v1.gfile.MakeDirs(ckpt_dir)

    run_config = tf.estimator.RunConfig(
        model_dir=ckpt_dir,
        save_checkpoints_steps=save_checkpoints_steps)

    if vocab_filename is not None:
        with open(vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)

    # Build model
    model_fn = model_fn_builder(
        pegi_config=pegi_config,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu,
        item_size=item_size)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
                "batch_size": batch_size,
                "max_dec_len": max_dec_len,
                "beam_size": beam_size
            })

    if training:
        print("***** Running training *****")
        print("  Batch size = %d" % batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_input_file,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            training=True,
            batch_size=batch_size)

        print("\n\n####### before estimator.train #######\n\n")
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        print("\n\n####### after estimator.train #######\n\n")

    print("***** Running evaluation *****")
    print("  Batch size = %d" % batch_size)
    # training=False for eval. Controls whether dropout will be applied.
    # Rebuild input pipeline
    pred_input_fn = input_fn_builder(
                            input_files=test_input_file,
                            max_seq_length=max_seq_length,
                            max_predictions_per_seq=max_predictions_per_seq,
                            training=False,
                            batch_size=batch_size)

    inputs = pred_input_fn()
    features = tf.data.experimental.get_single_element(inputs.take(1))

    # Count the number of iteration
    cnt = inputs.reduce(np.int64(0), lambda x, _: x + 1)
    with tf.compat.v1.Session() as sess:
        cnt = sess.run([cnt])
        
    # Rebuild the model
    prediction = model_fn(features, None,
                        mode=tf.estimator.ModeKeys.EVAL,
                        params={
                                "batch_size": batch_size,
                                "max_dec_len": max_dec_len,
                                "beam_size": beam_size}).predictions

    # Initialize for new variables
    init_op = tf.compat.v1.global_variables_initializer()
    ct = 0
    with tf.compat.v1.train.MonitoredSession(hooks=[EvalHooks()]) as sess:
        sess.run(init_op)
        while True:
            sess.run([prediction, features])
            ct += 1
            if ct == cnt[0]:
                print("done")
                break



if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.enable_resource_variables()
    tf.compat.v1.app.run()


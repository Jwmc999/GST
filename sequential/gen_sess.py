import os
import pickle

import random
import time
import multiprocessing
import collections
import tensorflow as tf

from utils.util import *
from vocab import *

dataset_name = "session_ml-1m"
max_seq_length = 20
gap_sg_prob = 0.2
max_predictions_per_seq = 2

prop_sliding_window = 0.5
mask_prob = 1.0
dupe_factor = 10
pool_size = 10

version_id = ''
output_dir = 'data_dir/'
random_seed = 12345
short_seq_prob = 0
task = 'summarization'

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, gap_sg_positions, gap_sg_labels):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.gap_sg_positions = gap_sg_positions
        self.gap_sg_labels = gap_sg_labels

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([x for x in self.info]))
        s += "tokens: %s\n" % (
            " ".join([x for x in self.tokens]))
        s += "gap_sg_positions: %s\n" % (
            " ".join([str(x) for x in self.gap_sg_positions]))
        s += "gap_sg_labels: %s\n" % (
            " ".join([x for x in self.gap_sg_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    # TFRecordWriter()
    # 
    writers = []
    for output_file in output_files:
        writers.append(tf.compat.v1.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        try:
            input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        except:
            print(instance)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        gap_sg_positions = list(instance.gap_sg_positions)
        gap_sg_ids = vocab.convert_tokens_to_ids(instance.gap_sg_labels)
        gap_sg_weights = [1.0] * len(gap_sg_ids)

        gap_sg_positions += [0] * (max_predictions_per_seq - len(gap_sg_positions))
        gap_sg_ids += [0] * (max_predictions_per_seq - len(gap_sg_ids))
        gap_sg_weights += [0.0] * (max_predictions_per_seq - len(gap_sg_weights))

        features = collections.OrderedDict()
        features["label_info"] = create_int_feature(instance.info)
        features["output_ids"] = create_int_feature(input_ids)
        features["output_mask"] = create_int_feature(input_mask)
        features["gap_sg_positions"] = create_int_feature(
            gap_sg_positions)
        features["gap_sg_ids"] = create_int_feature(gap_sg_ids)
        features["gap_sg_weights"] = create_float_feature(gap_sg_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            print("*** Example ***")
            print("tokens: %s" % " ".join([x for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                print("%s: %s" % (feature_name," ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()
    print("Wrote %d total instances", total_written)
    
def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature

def create_training_instances(all_documents_raw,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              gap_sg_prob,
                              max_predictions_per_seq,
                              rng,
                              vocab,
                              mask_prob,
                              prop_sliding_window,
                              pool_size,
                              force_last=False):
    """Create `TrainingInstance`s from raw text."""
    all_documents = {}

    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue
            all_documents[user] = [item_seq[-max_num_tokens:]]
    else:
        max_num_tokens = max_seq_length  # we need two sentence
        
        sliding_step = (int)(
            prop_sliding_window *
            max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue

            # todo: add slide
            if len(item_seq) <= max_num_tokens:
                all_documents[user] = [item_seq]
            else:
                beg_idx = list(range(len(item_seq) - max_num_tokens, 0, -sliding_step))
                beg_idx.append(0)
                all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]

    instances = []
    if force_last:
        for user in all_documents:
            instances.extend(
                create_instances_from_document_test(
                    all_documents, user, max_seq_length))
        print("num of instance:{}".format(len(instances)))
    else:
        start_time = time.time()
        pool = multiprocessing.Pool(processes=pool_size)
        instances = []
        print("document num: {}".format(len(all_documents)))

        def log_result(result):
            print("callback function result type: {}, size: {} ".format(type(result), len(result)))
            instances.extend(result)

        for step in range(dupe_factor):
            pool.apply_async(
                create_instances_threading, args=(
                    all_documents, max_seq_length, short_seq_prob,
                    gap_sg_prob, max_predictions_per_seq, vocab, random.Random(random.randint(1, 10000)),
                    mask_prob, step), callback=log_result)
        pool.close()
        pool.join()

        for user in all_documents:
            instances.extend(
                mask_last(
                    all_documents, user, max_seq_length, short_seq_prob,
                    gap_sg_prob, max_predictions_per_seq, vocab, rng))

        print("num of instance:{}; time:{}".format(len(instances), time.time() - start_time))
    rng.shuffle(instances)
    return instances

def create_instances_threading(all_documents, max_seq_length, short_seq_prob,
                               gap_sg_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step):
    cnt = 0
    start_time = time.time()
    instances = []
    for user in all_documents:
        cnt += 1
        if cnt % 1000 == 0:
            print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt,
                                                                  time.time() - start_time))
            start_time = time.time()
        instances.extend(create_instances_from_document_train(
            all_documents, user, max_seq_length, short_seq_prob,
            gap_sg_prob, max_predictions_per_seq, vocab, rng,
            mask_prob))
    return instances

# mask last for next item prediction
def mask_last(
        all_documents, user, max_seq_length, short_seq_prob, gap_sg_prob,
        max_predictions_per_seq, vocab, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

        (tokens, gap_sg_positions,
         gap_sg_labels) = create_gap_sg_predictions_force_last(tokens)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            gap_sg_positions=gap_sg_positions,
            gap_sg_labels=gap_sg_labels)
        instances.append(instance)
    return instances

def create_instances_from_document_test(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    assert len(document) == 1 and len(document[0]) <= max_num_tokens

    tokens = document[0]
    assert len(tokens) >= 1
    
    if task == "summarization":
        (tokens, gap_sg_positions,
        gap_sg_labels) = create_gap_sg_predictions_no_mask(tokens)
    else:
        (tokens, gap_sg_positions,
        gap_sg_labels) = create_gap_sg_predictions_force_last(tokens)

    info = [int(user.split("_")[1])]
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        gap_sg_positions=gap_sg_positions,
        gap_sg_labels=gap_sg_labels)
    return [instance]

def create_instances_from_document_train(
        all_documents, user, max_seq_length, short_seq_prob, gap_sg_prob,
        max_predictions_per_seq, vocab, rng, mask_prob):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]

    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

        (tokens, gap_sg_positions,
         gap_sg_labels) = create_gap_sg_predictions(
            tokens, gap_sg_prob, max_predictions_per_seq,
            vocab_items, rng, mask_prob)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            gap_sg_positions=gap_sg_positions,
            gap_sg_labels=gap_sg_labels)
        instances.append(instance)
    return instances

# Masked LM
GapSGInstance = collections.namedtuple("GapSGInstance", ["index", "label"])

def create_gap_sg_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
            continue
        last_index = i

    assert last_index > 0

    output_tokens = list(tokens)
    output_tokens[last_index] = "[MASK]"

    gap_sg_positions = [last_index]
    gap_sg_labels = [tokens[last_index]]
    return (output_tokens, gap_sg_positions, gap_sg_labels)

def create_gap_sg_predictions_no_mask(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
            continue
        last_index = i

    assert last_index > 0

    output_tokens = list(tokens)

    gap_sg_positions = [last_index]
    gap_sg_labels = [tokens[last_index]]
    return (output_tokens, gap_sg_positions, gap_sg_labels)

def create_gap_sg_predictions(tokens, gap_sg_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token not in vocab_words:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * gap_sg_prob))))

    gap_sgs = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(gap_sgs) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < mask_prob:
            masked_token = "[MASK]"
        # # if ndcg score is high, replace with [MASK]
        # if sess_ndcg > 0.1 or sess_hit > 0.1:
        #     masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                masked_token = rng.choice(vocab_words)

        output_tokens[index] = masked_token

        gap_sgs.append(GapSGInstance(index=index, label=tokens[index]))

    gap_sgs = sorted(gap_sgs, key=lambda x: x.index)

    gap_sg_positions = []
    gap_sg_labels = []
    for p in gap_sgs:
        gap_sg_positions.append(p.index)
        gap_sg_labels.append(p.label)
    return (output_tokens, gap_sg_positions, gap_sg_labels)

# Generate samples
def gen_samples(data,
                output_filename,
                rng,
                vocab,
                max_seq_length,
                dupe_factor,
                short_seq_prob,
                mask_prob,
                gap_sg_prob,
                max_predictions_per_seq,
                prop_sliding_window,
                pool_size,
                force_last=False):
    # create train
    instances = create_training_instances(
        data, max_seq_length, dupe_factor, short_seq_prob, gap_sg_prob,
        max_predictions_per_seq, rng, vocab, mask_prob, prop_sliding_window,
        pool_size, force_last)

    write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    [output_filename])


def main():
    os.makedirs(output_dir+task, exist_ok=True)
    dataset = data_partition(output_dir + dataset_name + '.txt', mode='sess')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    cc = 0.0
    max_len = 0
    min_len = 100000
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(len(user_train[u]), max_len)
        min_len = min(len(user_train[u]), min_len)

    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('max:{}, min:{}'.format(max_len, min_len))

    print('len_train:{}, len_valid:{}, len_test:{}, usernum:{}, itemnum:{}'.
        format(
        len(user_train),
        len(user_valid), len(user_test), usernum, itemnum))

    for idx, u in enumerate(user_train):
        if idx < 10:
            print(user_train[u])
            print(user_valid[u])
            print(user_test[u])

    # put validate into train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])

    # get the max index of the data
    user_train_data = {
        'user_' + str(k): ['sess_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    user_test_data = {
        'user_' + str(u):
            ['sess_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }
    rng = random.Random(random_seed)

    vocab = FreqVocab(user_test_data)
    user_test_data_output = {
        k: [vocab.convert_tokens_to_ids(v)]
        for k, v in user_test_data.items()
    }

    print('begin to generate train')
    output_filename = output_dir + task + '/' + dataset_name + version_id + '.train.tfrecord'
    gen_samples(
        user_train_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        gap_sg_prob,
        max_predictions_per_seq,
        prop_sliding_window,
        pool_size,
        force_last=False)
    print('train:{}'.format(output_filename))

    print('begin to generate test')
    """Finetuning input dataset
    
    Summarization task: create_gap_sg_predictions_no_mask
    Generation task: create_gap_sg_predictions_force_last
    """
    output_filename = output_dir + task + '/' + dataset_name + version_id + '.test.tfrecord'
    gen_samples(
        user_test_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        gap_sg_prob,
        max_predictions_per_seq,
        -1, 
        pool_size,
        force_last=True) 
    print('test:{}'.format(output_filename))
        
    print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
          format(vocab.get_vocab_size(),
                 vocab.get_user_count(),
                 vocab.get_item_count(),
                 vocab.get_item_count() + vocab.get_special_token_count()))
    vocab_file_name = output_dir + task + '/' + dataset_name + version_id + '.vocab'
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)

    his_file_name = output_dir + task + '/' + dataset_name + version_id + '.his'
    print('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    print('done.')

if __name__ == '__main__':
    main()
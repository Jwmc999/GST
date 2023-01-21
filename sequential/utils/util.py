from collections import defaultdict
import ast
import tensorflow as tf

# Originally from util.py of BERT4rec
# Divide dataset into train, test, validate.
# Also counts number of users and items 
def data_partition(fname, mode=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open(fname, 'r') as f:
        lines = list(f.readlines())
        for line in lines:
            line = line.rstrip()
            if ' ' not in line:
                continue
            if mode == 'sess':
                line_split = line.split(' ', 1)
                u = int(line_split[0])
                sess = ast.literal_eval(line_split[1])
                usernum = max(u, usernum)
                itemnum = max(len(sess), itemnum)
                User[u].append(sess)
            else:
                line_split = line.rsplit(' ')
                u = int(line_split[0])
                i = int(line_split[1])
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback >= 3:
                user_train[user] = (User[user][:-2])
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
            else: # encoder - decoder requires same document length
                user_train[user] = User[user]
                user_valid[user] = []
                user_valid[user].append(['NO_USE'])
                user_test[user] = []
                user_test[user].append(['NO_USE'])

    return [user_train, user_valid, user_test, usernum, itemnum]

## Copied from pegasus.pegasus.data.utils.py
"""Utils for parsers.

Shape notations:
U: unknown dimensions, L: length,
B: batch_size, I: max_input_length, T: max_target_length.
"""
def filter_by_length(tensor_list, min_len_list=None, max_len_list=None):
  """Filter tensors by their minimum or maximum length."""
  if not min_len_list and not max_len_list:
    return tensor_list
  if min_len_list:
    if len(min_len_list) != len(tensor_list):
      raise ValueError("Min length list need to match size of tensor_list.")
  else:
    min_len_list = [None for _ in tensor_list]

  if max_len_list:
    if len(max_len_list) != len(tensor_list):
      raise ValueError("Max length list need to match size of tensor_list.")
  else:
    max_len_list = [None for _ in tensor_list]

  keep = tf.constant(True, dtype=tf.bool)
  for min_len, max_len, tensor in zip(min_len_list, max_len_list, tensor_list):
    if min_len and max_len and min_len >= max_len:
      raise ValueError("Invalid min max lengths.")
    if any([min_len, max_len]):
      tensor_len = tf.reduce_sum(tf.cast(tf.greater(tensor, 0), tf.int32))
      if min_len:
        keep = tf.logical_and(keep, tf.greater(tensor_len, min_len))
      if max_len:
        keep = tf.logical_and(keep, tf.less_equal(tensor_len, max_len))

  filtered_tensor_list = []
  for tensor in tensor_list:
    empty_tensor = tf.zeros(
        [0] * len(tensor.shape.as_list()), dtype=tensor.dtype)
    filtered_tensor_list.append(
        tf.cond(keep, lambda: tensor, lambda: empty_tensor))  # pylint: disable=cell-var-from-loop
  return filtered_tensor_list


def add_length_bucket_id(inputs_BxI, targets_BxT, bucket_size, bucket_start_id,
                         max_num_buckets):
  """Add bucket id of the target to start of the inputs."""
  if bucket_size:
    non_pad_BxL = tf.cast(tf.greater(targets_BxT, 0), targets_BxT.dtype)
    length_Bx1 = tf.reduce_sum(non_pad_BxL, axis=-1, keep_dims=True)
    bucket_id_Bx1 = length_Bx1 // bucket_size + bucket_start_id
    # tail distributions are assigned to the last bucket.
    bucket_id_Bx1 = tf.minimum(bucket_id_Bx1, max_num_buckets)
    inputs_BxI = tf.concat([bucket_id_Bx1, inputs_BxI[:, :-1]], axis=-1)
  return inputs_BxI


def add_task_id(inputs_1xI, task_id):
  task_id_1x1 = tf.cast(tf.reshape(task_id, [1, 1]), inputs_1xI.dtype)
  return tf.concat([task_id_1x1, inputs_1xI[:, :-1]], axis=1)
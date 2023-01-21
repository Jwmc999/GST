import numpy as np
from scipy.sparse import csr_matrix
import datetime

from collections import Counter
import torch
import torch.nn.functional as F
import math
from functools import reduce

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    return matrix

def mask_with_session(t, session_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), [session_ids], init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

def flatten_lists(the_lists):
    result = []
    for _list in the_lists:
        result += _list
    return result

def reorder(train, test, rank):
    print("data remapping start: ", datetime.datetime.now())
    train_d = train[0].copy()
    test_d = test[0].copy()
    train_y = train[1].copy()
    test_y = test[1].copy()
    raw_d = train_d + test_d
    theme = Counter(flatten_lists(raw_d)).most_common(rank)
    theme = [tu[0] for tu in theme]
    corpus = [session for t in theme for session in raw_d if t in session]
    whole_d = list(set(flatten_lists(corpus) + train_y + test_y))
    itemmap = dict()
    itemnum = 0
    #Remap item_id
    for asin in whole_d:
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
    for ls in range(len(train_d)):
        for t in range(len(train_d[ls])):
            if train_d[ls][t] in whole_d:
                train_d[ls][t] = itemmap[train_d[ls][t]]
                continue
            train_d[ls][t] = None
    for ls in range(len(test_d)):
        for t in range(len(test_d[ls])):
            if test_d[ls][t] in whole_d:
                test_d[ls][t] = itemmap[test_d[ls][t]]
                continue
            test_d[ls][t] = None
    train_d = [sublist for sublist in train_d if all(sublist)]
    test_d = [sublist for sublist in test_d if all(sublist)]
    for ls in range(len(train_y)):
        train_y[ls] = itemmap[train_y[ls]]
    for ls in range(len(test_y)):
        test_y[ls] = itemmap[test_y[ls]]
    train = [train_d, train_y]
    test = [test_d, test_y]
    print("data remapping done: ", datetime.datetime.now())
    return train, test, len(itemmap)

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, rng, 
                 max_seq_len,
                 shuffle=False, 
                 n_pred=None,
                 mask_prob=1.0,
                 gap_sg_prob=0.15,
                 ):
        self.raw = np.asarray(data[0])        
        num_to_predict = min(n_pred,
                         max(1, int(round(len(data) * gap_sg_prob))))

        self.targets = np.asarray(data[1])
        self.shuffle = shuffle
        self.mask_prob = mask_prob
        self.gap_sg_prob = gap_sg_prob
        self.mask_token_id = 2
        self.pad_token_id = 0
        self.mask_ignore_token_ids = 3
        self.rng = rng
        self.num_tokens = num_to_predict
        self.vocab_size = 0
        self.max_seq_len = max_seq_len
        self.length = len(self.raw) 

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            self.rng.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        return self.targets[index]-1, session_len,items, reversed_sess_item, mask
        
    def get_batch(self, index):
        inp = self.raw[index]
        t = []
        for session in inp:
            session = torch.as_tensor(np.array(session), dtype=torch.int64)
            if len(session) < 2:
                session = F.pad(session, pad=(0, self.max_seq_len - len(session)), 
                    mode='constant', value=self.mask_ignore_token_ids)
            else:
                session = F.pad(session, pad=(0, self.max_seq_len - len(session)), 
                    mode='constant', value=self.pad_token_id)
            t.append(session)
        t = torch.stack(t)
        no_mask = mask_with_session(t, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.gap_sg_prob)
        masked_seq = t.clone().detach()
        labels = t.masked_fill(~mask, self.pad_token_id)
        if self.rng.random() > 0:
            assert self.num_tokens is not None, 'num_tokens required for GSG random masking initialization'
            random_token_prob = prob_mask_like(t, self.rng.random())
            random_tokens = torch.randint(0, self.num_tokens, t.shape, device=t.device)
            random_no_mask = mask_with_session(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)
            mask = mask & ~random_token_prob
        mask_prob = prob_mask_like(t, self.mask_prob)
        masked_seq = masked_seq.masked_fill(mask * mask_prob, self.mask_token_id)
        return masked_seq, labels, self.pad_token_id
    
    
class Load(torch.utils.data.Dataset):
    def __init__(self, data, rng, 
                 max_seq_len,
                 shuffle=False, 
                 n_pred=None,
                 mask_prob=1.0,
                 gap_sg_prob=0.15,
                 ):
        self.raw = np.asarray(data[0])
        num_to_predict = min(n_pred,
                         max(1, int(round(len(data) * gap_sg_prob))))
        self.targets = np.asarray(data[1])
        self.shuffle = shuffle
        self.mask_prob = mask_prob
        self.gap_sg_prob = gap_sg_prob
        self.mask_token_id = 2
        self.pad_token_id = 0
        self.mask_ignore_token_ids = 3
        self.rng = rng
        self.num_tokens = num_to_predict
        self.vocab_size = 0
        self.max_seq_len = max_seq_len
        self.length = len(self.raw) 
    
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        inp = self.raw[index]
        inp = torch.as_tensor(np.array(inp), dtype=torch.int64)
        if len(inp) < 2:
            session = F.pad(inp, pad=(0, self.max_seq_len - len(inp)), 
                mode='constant', value=self.mask_ignore_token_ids)
        else:
            session = F.pad(inp, pad=(0, self.max_seq_len - len(inp)), 
                mode='constant', value=self.pad_token_id)
        t = session.clone()
        no_mask = mask_with_session(t, self.mask_ignore_token_ids)
        mask = _get_mask_subset_with_prob(~no_mask, self.gap_sg_prob)
        masked_seq = t.clone().detach()
        labels = self.targets[index]-1
        if self.rng.random() > 0:
            assert self.num_tokens is not None, 'num_tokens required for GSG random masking initialization'
            random_token_prob = prob_mask_like(t, self.rng.random())
            random_tokens = torch.randint(0, self.num_tokens, t.shape, device=t.device)
            random_no_mask = mask_with_session(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)
            mask = mask & ~random_token_prob
        mask_prob = prob_mask_like(t, self.mask_prob)
        masked_seq = masked_seq.masked_fill(mask * mask_prob, self.mask_token_id)
        return masked_seq, labels
    
    
def _get_mask_subset_with_prob(mask, prob):
    seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:max_masked]
    rand = torch.rand((seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[1:].bool()
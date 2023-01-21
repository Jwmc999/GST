#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='nowplaying', help='dataset name: diginetica/yoochoose/nowplaying')
opt = parser.parse_args()
print(opt)

data_dir = "archive/"
if opt.dataset == 'diginetica':
    dataset = data_dir + 'train-item-views.csv'
elif opt.dataset =='yoochoose':
    dataset = data_dir + 'yoochoose-clicks.dat'
else:
    dataset = data_dir + 'nowplaying.csv'



print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',',
                                fieldnames = ("session_id", "timestamp", 
                                              "item_id",  "price", 
                                              "quantity"))
    elif opt.dataset == 'diginetica':
        reader = csv.DictReader(f, delimiter=';', 
                                fieldnames = ("session_id", "user_id", 
                                              "item_id",  "timeframe", 
                                              "eventdate"))
        next(reader)
    else:
        reader = csv.DictReader(f, delimiter='\t',
                                fieldnames = ("user_id", "session_id", 
                                              "item_id", "time",
                                              "artist"))
        next(reader)
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            elif opt.dataset == 'nowplaying':
                date = int(curdate)
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset in ['yoochoose', 'nowplaying']:
            item = data['item_id']
        else:
            item = data['item_id'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        elif opt.dataset == 'diginetica':
            curdate = data['eventdate']
        else:
            curdate = data['time']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    elif opt.dataset == 'nowplaying':
        date = int(curdate)
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        # sess_clicks[s] = 'NO_USE'
        # sess_date[s] = 'NO_USE'
        del sess_clicks[s]
        del sess_date[s]



# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print("item ctr", item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('data_dir/diginetica'):
        os.makedirs('data_dir/diginetica')
    pickle.dump(tra, open('data_dir/diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('data_dir/diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('data_dir/diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('data_dir/yoochoose'):
        os.makedirs('data_dir/yoochoose')
    pickle.dump(tes, open('data_dir/yoochoose/test.txt', 'wb'))
    split64 = int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split64:]))
    tra64 = (tr_seqs[-split64:], tr_labs[-split64:])
    seq64 = tra_seqs[tr_ids[-split64]:]
    pickle.dump(tra64, open('data_dir/yoochoose/train.txt', 'wb'))
    pickle.dump(seq64, open('data_dir/yoochoose/all_train_seq.txt', 'wb'))
else:
    if not os.path.exists('data_dir/nowplaying'):
        os.makedirs('data_dir/nowplaying')
    pickle.dump(tra, open('data_dir/nowplaying/train.txt', 'wb'))
    pickle.dump(tes, open('data_dir/nowplaying/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('data_dir/nowplaying/all_train_seq.txt', 'wb'))

print('Done.')
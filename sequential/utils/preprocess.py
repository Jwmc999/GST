# -*- coding: utf-8 -*-

# Data Preprocessing for item/session masking
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def dat2csv(file_path, dataset_name):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    
    if dataset_name.endswith('.dat'):
      df = pd.read_csv(file_path + '/' + dataset_name, sep='::', 
                     header=None, names=names, engine='python')
    elif dataset_name.endswith('.csv'):
      if dataset_name == ("anime.csv"):
        df = pd.read_csv(file_path + '/' + dataset_name, index_col=False, engine='python')  
        df = df[['username', 'anime_id', 'my_score', 'my_last_updated']]
        df.rename(columns = {'username':names[0], 'anime_id':names[1], 'my_score':names[2], \
                             'my_last_updated':names[3]}, inplace = True)
        labelencoder = LabelEncoder()
        df['user_id'] = labelencoder.fit_transform(df['user_id'])
      else:
        df = pd.read_csv(file_path + '/' + dataset_name, 
                      header=None, names=names, engine='python')          
    return df

def reorder(imp_df):
  """Following BERT4rec, we adapted data preprocessing from SASRec.
  This remaps user-item interaction sequence by timestamp.
  
  Original paper:
  https://ieeexplore.ieee.org/document/8594844
  
  Original code:
  https://github.com/kang205/SASRec

  """
  # rating to binary (1, 0)
  imp_df['rating'] = imp_df['rating'].map(lambda x: 0 if (x==0) else 1)  
  # keep only rating ==1
  imp_df = imp_df[imp_df.rating != 0]
    
  a = imp_df['item_id'].tolist()
  r = imp_df['user_id'].tolist()
  ts = imp_df['timestamp'].tolist()

  countU = defaultdict(lambda: 0)
  countP = defaultdict(lambda: 0)  
  
  for asin, rev, _ in zip(a, r, ts):
    countU[rev] += 1
    countP[asin] += 1
    
  usermap = dict()
  usernum = 0
  itemmap = dict()
  itemnum = 0
  User = dict()

  # Interaction sequence
  for asin, rev, t in zip(a, r, ts):
    
    # User should have at least five feedbacks.
    if countU[rev] < 5:
      continue
    ## for ml-1m, item should also have at least five feedbacks
    # if countU[rev] < 5 or countP[asin] < 5: 
    #   continue
    
    #Remap user_id
    if rev in usermap:
      userid = usermap[rev]
    else:
      usernum += 1
      userid = usernum
      usermap[rev] = userid
      User[userid] = []  
      
    #Remap item_id
    if asin in itemmap:
      itemid = itemmap[asin]
    else:
      itemnum += 1
      itemid = itemnum
      itemmap[asin] = itemid
    User[userid].append([t, itemid])

  print(usernum, itemnum)

  # Interaction sequence for each user
  # sort items in User according to time
  for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])
  return User  

def n_gram(sess, gram):
  if gram == 0:
    gram = 2
  chunked = [sess[i:i + gram] for i in range(0, len(sess), gram)]
  return chunked
  
def _item(imp_df, name):
  # User behavior 
  User = reorder(imp_df)
  
  dataset_names = os.path.splitext(name)[0]
  print(dataset_names)  
  
  # store the result in ml-1m.txt/mind.txt/idbm.txt file
  f = open(data_dir+'/'+dataset_names+'.txt', 'w')
  for user in User.keys():
    for i in User[user]:
      f.write('%d %d\n' % (user, i[1]))
  f.close()

def _session(imp_df, name, gram=None):
  # Sessioning User bahavior
  User = reorder(imp_df)

  Sess = dict()
  avg = dict()
  
  # Interaction sequence for each user
  # compute absolute differences between consecutive items
  for id in User.keys():
    t = [datetime.fromtimestamp(round(int(x)/1000)) for x, _ in User[id]]

    # time interval between each item
    interval = [abs(j-i) for i, j in zip(t[:-1], t[1:])]

    # User's time interval between interaction
    av = [a.total_seconds() for a in interval]
    avg[id] = (np.median(av) + np.min(av)) / 3   # np.percentile(av, 10) 
    # avg[id] = np.mean((np.median(av) + np.min(av))) 
    
    # Divide sequence into session      
    Sess[id] = [[User[id][0][1]]]   
    for i, j in zip(interval, range(1, len(interval)+1)):
        if i.total_seconds() <= avg[id]:
            Sess[id][-1].append(User[id][j][1])
        else:
            Sess[id].append([User[id][j][1]])            
            
  dataset_names = os.path.splitext(name)[0]
  print(dataset_names)

  # For pretraining, we need normalized sessions of user sequence.
  # In order to find out normalized time interval to sessionized,
  # we compute sum of the standard deviation of session length by 
  # user.  
  std = []
  mean = []
  for user in Sess.keys():
    sess = []
    for i in Sess[user]:
      # session length 
      sess.append(len(i))
    # standard deviation of session length
    std.append(np.std(sess))
    # mean of session length
    mean.append(np.median(sess))
  # sum/std of standard deviation of session length
  print('sum: {0:.3f} std: {1:.3f} mean: {2:.3f}'.format(np.sum(std), np.std(std), np.mean(mean)))

  # store the result in session_ml-1m.txt/ml-20m.txt/idbm.txt file
  f = open(data_dir+'/session_'+dataset_names+'.txt', 'w')
  for user in Sess.keys():
    # Divide session by n_gram, 
    # which is a half of median session length
    for item in Sess[user]:
      if len(item) > np.mean(mean):
        gram = int(np.mean(mean) / 2)
        item = n_gram(item, gram=gram)
        for j in item:
          f.write("{} {}\n".format(user, j))
      else:
        f.write('{} {}\n'.format(user, item)) 
  f.close()


if __name__ == '__main__':
  INPUT_DIR = 'archive'
  data_dir='data_dir'
  dataset_name = ['ml-1m.dat', 'ml-20m.csv', 'anime.csv']
  data1 = dat2csv(INPUT_DIR, dataset_name[0])
  data2 = dat2csv(INPUT_DIR, dataset_name[1])
  data3 = dat2csv(INPUT_DIR, dataset_name[2])
  # _item(data1, dataset_name[0])
  # _session(data1, dataset_name[0])
  # _item(data2, dataset_name[1])  
  # _session(data2, dataset_name[1]) 
  # _item(data3, dataset_name[2])
  _session(data3, dataset_name[2])
import re
import sqlite3
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

""" Custom Libs """
import Cleaner as c

seed = 666 #For reproducibility
np.random.seed(seed)

def fetch_profiles(filename, n):
    f           = open(filename, 'r')
    profiles    = f.read().splitlines()
    f.close()
    return(list(set(profiles[:n])))

#Import Data
sqlite_file = '../../data/database/deeplearning.sqlite'
profilename = '../../data/profiles.txt'
table_name  = 'tweets'
profiles    = fetch_profiles(profilename, 4)
profiles    = [p.strip('@') for p in profiles]
cd          = c.CleanData(sqlite_file, table_name)
q           = 'SELECT * FROM {} WHERE AUTHOR IN ("{}");'.format(table_name, '", "'.join(profiles))


cd.set_table(q)
raw_data = cd.get_clean_table()
raw_data = raw_data.CleanText.values
data = ''
for x in raw_data:
    data += x + "\n"

sentences = [x.split(' ') for x in raw_data]

train, test = train_test_split(sentences, test_size = 0.2)
comments = train + test

word_model = Word2Vec(
    comments,
    size = 150,
    sg = 0,
    window = 5,
    min_count = 2,
    workers = 10)
word_model.train(comments, total_examples=len(comments), epochs = 200)

pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape

word_model.save("word2vec.model")

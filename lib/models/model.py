import re
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense, Dropout, Activation, AveragePooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback
from keras.models import Sequential
import keras.utils as ku

from gensim.models import Word2Vec

""" Custom Libs """
import Cleaner as c

#Hyperparameters

epochs      = 300
dropout     = 0.2
l2_reg      = 1e-4
batch_sz    = 32
learn_rate  = 1e-3
beta_1      = 0.9
beta_2      = 0.999
epsilon     = None
decay_rate  = 0
amsgrad     = False
run_model   = True
seed        = 777 #For reproducibility

opt_adam = Adam(lr = learn_rate, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, decay = decay_rate, amsgrad = amsgrad)

def fetch_profiles(filename, n):
    f           = open(filename, 'r')
    profiles    = f.read().splitlines()
    f.close()
    return(list(set(profiles[:n])))

sqlite_file = '../../data/database/deeplearning.sqlite'
profilename = '../../data/profiles.txt'
table_name  = 'tweets'
profiles    = fetch_profiles(profilename, 15)
profiles    = [p.strip('@') for p in profiles]
cd          = c.CleanData(sqlite_file, table_name)
q           = 'SELECT * FROM {} WHERE AUTHOR IN ("{}");'.format(table_name, '", "'.join(profiles))

word_model = Word2Vec.load("word2vec.model")

np.random.seed(seed)

def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

cd.set_table(q)
raw_data = cd.get_clean_table()
raw_data = raw_data.CleanText.values
data = ''
for x in raw_data:
    data += x + "\n"

sentences = [x.split(' ') for x in raw_data]

allowed_words = set(list(word_model.wv.vocab))
comp = [x.split(' ') for x in raw_data]
sentences = []
for l in comp:
    t = [x for x in l if x in allowed_words]
    sentences.append(t)
sentences = [x for x in sentences if x != []]

max_sentence_len = max([len(sentence) for sentence in sentences])

train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(sentence[-1])


def sample(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature = 0.8)
    word_idxs.append(idx)
    if idx2word(idx) == '.':
        return ' '.join(idx2word(idx) for idx in word_idxs)
  return ' '.join(idx2word(idx) for idx in word_idxs)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    'data',
    'deep',
    'machine',
    'business'
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))

def create_model(weights):
    model = Sequential()
    model.add(Embedding(input_dim = weights.shape[0], output_dim = weights.shape[1], weights = [weights]))
    model.add(LSTM(256, return_sequences = True))
    if dropout != 0:
        model.add(Dropout(dropout))
        model.add(LSTM(512))
    else:
        model.add(LSTM(512))
    if l2_reg != 0:
        model.add(Dense(weights.shape[0], activation = 'softmax', bias_regularizer = l2(l2_reg)))
    else:
        model.add(Dense(weights.shape[0], activation = 'softmax'))

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt_adam, metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='model'
                                   + '/single-user-model-{}'.format(time()) + '-{epoch:02d}.hdf5', verbose = 1)
    tensorboard = TensorBoard(log_dir = 'tb-logs/{}'.format(time()))
    earlystop = EarlyStopping(monitor = 'loss', min_delta = 0, patience = 100, verbose = 0, mode = 'min')
    return(model, checkpointer, tensorboard, earlystop)

#word_model = Word2Vec.load("word2vec.model")
pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape


model, checkpointer, tensorboard, earlystop = create_model(pretrained_weights)
print(model.summary())

if run_model == True:
    if tf.test.is_gpu_available():
        model.fit(x = train_x, y = train_y,
                  epochs = epochs,
                  batch_size = batch_sz,
                  validation_split = 0.20,
                  verbose = 1,
                  callbacks=[checkpointer, tensorboard, earlystop, LambdaCallback(on_epoch_end=on_epoch_end)])

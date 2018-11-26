import tensorflow as tf
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation, Softmax
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
import sqlite3
import re
import sys

""" PARAMS """
VOCAB_L = 20
DROPOUT = 0.40
HIDDEN  = 128
BATCH   = 50
N_EPOCH = 35


sqlite_file = '../../data/database/deeplearning.sqlite'
table_name  = 'tweets'
cnxn = sqlite3.connect(sqlite_file)
q    ='SELECT * FROM {};'.format(table_name)
data = pd.read_sql_query(q, cnxn)

def strip_links(txt):
  txt = re.sub(r'(?:\w+|\@\w+|\#\w+)\.twitter\.com\/\w+', '', txt)
  return(re.sub(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', txt))

def strip_whitespace(txt):
  txt = txt.strip(' ')
  return(re.sub(r' +', ' ', txt))

def strip_metachar(txt):
  return(re.sub(r"[^a-zA-Z0-9\-\@\#\.\, ]+", '', txt))

def strip_ats(txt):
  return(re.sub(r'\@\w*', '', txt))

data['CleanText'] = data['Text'].apply(lambda t: strip_links(t))
data['CleanText'] = data['CleanText'].apply(lambda t: strip_whitespace(t))
data['CleanText'] = data['CleanText'].apply(lambda t: strip_metachar(t))
data['CleanText'] = data['CleanText'].apply(lambda t: strip_ats(t))
raw_text = ""
for tweet in data.CleanText:
 raw_text += tweet.strip()
 raw_text += ' '

raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = VOCAB_L
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(HIDDEN, input_shape=(X.shape[1], X.shape[2]))) # Add extra LSTM layer
model.add(Dropout(DROPOUT))
model.add(Dense(y.shape[1], activation='softmax')) # Unidirectional
#model.add(Activation('softmax')) # Need activation
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="model/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5" # Write to folder, rather than puking all over my directory
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X, y, epochs=N_EPOCH, batch_size=BATCH, callbacks=callbacks_list)

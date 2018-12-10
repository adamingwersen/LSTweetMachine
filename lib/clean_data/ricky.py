import re
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np

with tf.device('/gpu:0'):


  """ Custom Libs """
  import Cleaner as c

  #Read tweets

  sqlite_file = 'deeplearning.sqlite'
  table_name  = 'tweets'
  cd          = c.CleanData(sqlite_file, table_name)
  q           ='SELECT * FROM {};'.format(table_name)

  cd.set_table(q)
  data = cd.get_clean_table()
  data = data.CleanText.values

  data_2 = ''
  for x in data:
      data_2 += x + "\n"
  data = data_2

  np.random.seed(0)

  tokenizer = Tokenizer()

  def dataset_preparation(data):

    # basic cleanup
    corpus = data.lower().split("\n")

    # tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
      token_list = tokenizer.texts_to_sequences([line])[0]
      for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words

  def create_model(predictors, label, max_sequence_len, total_words):

    model = Sequential()
    model.add(Embedding(total_words, 500, input_length=max_sequence_len-1))
    model.add(LSTM(1000, return_sequences = True))
    model.add(Dropout(0.1))
    model.add(LSTM(500))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='min')
    model.fit(predictors, label, epochs=150, verbose=1, callbacks=[earlystop])
    print (model.summary())
    return model

  def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
      token_list = tokenizer.texts_to_sequences([seed_text])[0]
      token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
      predicted = model.predict_classes(token_list, verbose=0)

      output_word = ""
      for word, index in tokenizer.word_index.items():
        if index == predicted:
          output_word = word
          break
      seed_text += " " + output_word
    return seed_text


  predictors, label, max_sequence_len, total_words = dataset_preparation(data)
  model = create_model(predictors, label, max_sequence_len, total_words)

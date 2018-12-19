import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import Cleaner as c
import TokenMgmt as tm

sqlite_file = '../../data/database/deeplearning.sqlite'
table_name  = 'donald'
cd          = c.CleanData(sqlite_file, table_name)
q           ='SELECT * FROM {};'.format(table_name)

cd.set_table(q)
data = cd.get_clean_table()

inp_sequences, total_words = tm.get_sequence_of_tokens(list(data.CleanText.values))
predictors, label, max_sequence_len = tm.generate_padded_sequences(inp_sequences, total_words)


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 42, input_length=input_len))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(1024))
    #model.add(Dropout(0.3))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    checkpointer = ModelCheckpoint(filepath='model' + '/model-{epoch:02d}.hdf5', verbose=1)
    return(model, checkpointer)

model, checkpointer = create_model(max_sequence_len, total_words)
#model.summary()

model.fit(predictors, label, epochs=300, verbose=1, callbacks=[checkpointer])

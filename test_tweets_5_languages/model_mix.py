#machine learning model shell with TensorFlow

import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras.models import Sequential # Allows us to just add the layers
from keras.layers import Dense, Dropout, Flatten # Different types of layers
from keras.layers import Conv2D, MaxPooling2D, Activation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
#from sklearn.model_selection import train_test_split
#from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
import matplotlib.pyplot as plt


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['tweet'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))



#https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
#https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
def rnn_lstm_model_1(activation, input_shape, num_classes):
    model = Sequential()


#------ Example of TensorFlow use from earlier project ------
# Function to build a CNN model for the real-time classification on computer
def standard_cnn_model_uci(activation, input_shape, num_classes):
  model = Sequential()
  model.add(Conv2D(64, kernel_size=(5,1), activation=activation,\
                   input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2,1)))
  model.add(Conv2D(64, kernel_size=(5,1)))
  model.add(MaxPooling2D(pool_size=(2,1)))
  model.add(Conv2D(64, kernel_size=(5,1)))
  model.add(MaxPooling2D(pool_size=(2,1)))
  model.add(Conv2D(64, kernel_size=(3,9)))
  model.add(Flatten())
  model.add(Dropout(0.25))
  model.add(Dense(512))
  model.add(Dropout(0.5))
  model.add(Dense(30))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  return model

# Function to train the model
def train(model, x_train, y_train, x_val, y_val, epochs, batch_size):
  es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01,\
                                     patience=240, mode='min',\
                                     restore_best_weights=True)
  model.compile(loss='categorical_crossentropy', optimizer='adam',\
                metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,\
            validation_data=(x_val, y_val), callbacks=[es])


#Code to actually run the model

batch_size = 1500
epochs = 350

model = standard_cnn_model_uci(activation='relu', \
                               input_shape=x_train[0].shape, num_classes=5)

# Compile the model and select loss-function, optimizer etc.
model.compile(loss='categorical_crossentropy', optimizer='adam', \
              metrics=['accuracy'])

# Fit the model to the data and specify batch size, epochs and validation data
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, \
                    verbose=1, validation_data=(x_val, y_val))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.axhline(y=0.9, color='r', linestyle='--')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#------------ End example -----------
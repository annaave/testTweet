from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


def run_lstm(vocab_size, embedding_dim, class_names, x_train_pad, y_train, x_validation_pad, y_validation,
             x_test_pad, y_test, x_test, tokenizer, num_epochs):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dense(embedding_dim, activation='relu'))
    model.add(Dense(len(class_names), activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=240, mode='min', restore_best_weights=True)

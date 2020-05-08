import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import fasttext


# Class to create and train a machine learning model
class TrainModel:
    def __init__(self):
        self.model = Sequential()

    def create_lstm_model(self, vocab_size, embedding_dim, class_names):
        self.model.add(Embedding(vocab_size, embedding_dim))
        self.model.add(Bidirectional(LSTM(embedding_dim)))
        self.model.add(Dense(embedding_dim, activation='relu'))
        self.model.add(Dense(len(class_names), activation='softmax'))
        self.model.summary()
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return self.model

    def create_train_fasttext_model(self, train_filename):
        self.model = fasttext.train_supervised(train_filename, min_count=1, minn=1, lr=0.5, epoch=30, ws=3,
                                               label_prefix='__label__', dim=50)
        print(self.model)

    def train_model(self, bat_size, num_epochs, x_train_pad, y_train, x_validation_pad, y_validation):
        history = self.model.fit(x_train_pad, y_train, batch_size=bat_size, epochs=num_epochs,
                                 validation_data=(x_validation_pad, y_validation), verbose=1)

        np.save('/home/myuser/testTweet/LID/saved_model/history_cleaned_data.npy', history.history)

    def save_model(self, name):
        file_path = '/home/myuser/testTweet/LID/saved_model_cleaned_data/' + name
        self.model.save(file_path)

    @staticmethod
    def tokenize(vocab_size, oov_tok, trunc_type, padding_type, max_length):
        train = pd.read_csv('training_data.csv')
        validation = pd.read_csv('validation_data.csv')
        test = pd.read_csv('test_data.csv')
        x_train = np.asarray([np.asarray(text) for text in train['tweets']])
        y_train = np.asarray([np.asarray(label) for label in train['language']])

        x_validation = np.asarray([np.asarray(text) for text in validation['tweets']])
        y_validation = np.asarray([np.asarray(label) for label in validation['language']])

        x_test = np.asarray([np.asarray(text) for text in test['tweets']])
        y_test = np.asarray([np.asarray(label) for label in test['language']])
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, char_level=True)
        # tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(train['tweets'])
        word_index = tokenizer.word_index
        print(dict(list(word_index.items())[0:10]))
        train_sequences = tokenizer.texts_to_sequences(x_train)
        x_train_pad = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        validation_sequences = tokenizer.texts_to_sequences(x_validation)
        x_validation_pad = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type,
                                         truncating=trunc_type)
        test_sequences = tokenizer.texts_to_sequences(x_test)
        x_test_pad = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        return x_train, x_train_pad, y_train, x_validation, x_validation_pad, y_validation, x_test, x_test_pad, y_test


def main():
    vocab_size = 300
    embedding_dim = 128
    num_epochs = 15
    bat_size = 32
    max_length = 150
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'

    class_names = ["English", "Swedish", "Spanish", "Portuguese", "Russian"]
    lstm_model = TrainModel()
    lstm_model.create_lstm_model(vocab_size, embedding_dim, class_names)
    x_train, x_train_pad, y_train, x_validation, x_validation_pad, y_validation, x_test, x_test_pad, y_test =\
        lstm_model.tokenize(vocab_size, oov_tok, trunc_type, padding_type, max_length)

    history = lstm_model.train_model(bat_size, num_epochs, x_train_pad, y_train, x_validation_pad, y_validation)
    lstm_model.save_model("lstm_model")


if __name__ == "__main__":
    main()
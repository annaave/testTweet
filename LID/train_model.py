import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional


# Class to create and train a Long-Short Term Memory machine learning model
class TrainModel:
    def __init__(self, model_type, embedding_dim, class_names, bat_size, num_epochs, vocab_size = 300):
        try:
            if model_type == 'LSTM':
                self.vocab_size = vocab_size
                self.embedding_dim = embedding_dim
                self.class_names = class_names
                self.bat_size = bat_size
                self.num_epochs = num_epochs
                self.model = Sequential()
                self.model.add(Embedding(vocab_size, embedding_dim))
                self.model.add(Bidirectional(LSTM(embedding_dim)))
                self.model.add(Dense(embedding_dim, activation='relu'))
                self.model.add(Dense(len(class_names), activation='softmax'))
                self.model.summary()
                self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
            else:
                raise ValueError('Invalid model type')
        except ValueError as exp:
            print('Only LSTM model type is valid at the moment.')

    def train_model(self, train_data, validation_data, history_path):
        history = self.model.fit(train_data['x_train_pad'], train_data['y_train'], batch_size=self.bat_size, epochs=self.num_epochs,
                                 validation_data=(validation_data['x_data_pad'], validation_data['y_data']), verbose=1)
        np.save(history_path, history.history)

    def save_model(self, name):
        file_path = '/home/myuser/testTweet/LID/saved_model/' + name
        self.model.save(file_path)

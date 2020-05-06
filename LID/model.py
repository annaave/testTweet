import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional
import sys
from LID.data import read_all


def create_model(vocab_size, embedding_dim, class_names):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dense(embedding_dim, activation='relu'))
    model.add(Dense(len(class_names), activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def train_model(model, bat_size, num_epochs, x_train_pad, y_train, x_validation_pad, y_validation):
    history = model.fit(x_train_pad, y_train, batch_size=bat_size, epochs=num_epochs,
                        validation_data=(x_validation_pad, y_validation), verbose=1)
    return history


def main():
    vocab_size = 300
    embedding_dim = 64
    num_epochs = 5

    class_names = ["English", "Swedish", "Spanish", "Portuguese", "Russian", "German"]
    all_data = read_all(class_names)
    all_data.to_csv("all_data_labels_random.csv", index=False)
    print(all_data)


if __name__ == "__main__":
    main()
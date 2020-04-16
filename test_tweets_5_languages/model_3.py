import pandas as pd
import numpy as np
import re
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding, Bidirectional
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

max_length = 150

def read_all(class_names):
    max_rows = 10000
    d = dict(zip(class_names, range(0, 5)))

    pd.options.display.max_colwidth = 1000

    df1 = (pd.read_csv("1000_Eng_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df1 = df1.drop(index=0)
    df1 = df1.reset_index(drop=True)

    df2 = (pd.read_csv("1000_Swe_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df2 = df2.drop(index=0)
    df2 = df2.reset_index(drop=True)

    df3 = (pd.read_csv("1000_Rus_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df3 = df3.drop(index=0)
    df3 = df3.reset_index(drop=True)

    df4 = (pd.read_csv("1000_Por_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df4 = df4.drop(index=0)
    df4 = df4.reset_index(drop=True)

    df5 = (pd.read_csv("1000_Spa_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df5 = df5.drop(index=0)
    df5 = df5.reset_index(drop=True)

    all_data = pd.concat([df1, df2, df3, df4, df5], ignore_index=True, sort=False)

    # Remove url:s
    for i in range(len(all_data)):
        all_data['tweets'][i] = clean_up(all_data['tweets'][i])

    all_data = all_data.sample(frac=1).reset_index(drop=True)
    all_data['language'] = all_data['language'].map(d, na_action='ignore')

    return all_data

def clean_up(line):
    # remove url
    line = re.sub(r'http\S+', '', line)
    return line

def split_data(frame):
    train, test = train_test_split(frame, test_size=0.2)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test

def encode_ngram_text(text, n):
    text = [text[i:i + n] for i in range(len(text) - n + 1)]
    text = [ord(text[i]) for i in range(len(text))]
    return text

def pad(data, length):
    new_data = []
    for row in data:
        new_row = np.pad(row, (0, length - len(row)), constant_values=0)
        new_data.append(new_row)
    return new_data
def main():
    class_names = ["English", "Swedish", "Spanish", "Portuguese", "Russian"]
    print("Classes:", class_names)

    all_data = read_all(class_names)

    print(set(all_data['language']))
    train, test = split_data(all_data)

    x_train = np.asarray([np.asarray(text) for text in train['tweets']])

    y_train = np.asarray([np.asarray(label) for label in train['language']])

    print(x_train)
    print(y_train)

    x_test = np.asarray([np.asarray(text) for text in test['tweets']])
    y_test = np.asarray([np.asarray(label) for label in test['language']])

    x_train = [encode_ngram_text(line, 1) for line in x_train]
    x_train = pad(x_train, max_length)
    x_test = [encode_ngram_text(line, 1) for line in x_test]
    x_test = pad(x_test, max_length)
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    y_train = np_utils.to_categorical(y_train, num_classes=5)
    y_test = np_utils.to_categorical(y_test, num_classes=5)


    print(len(x_train))
    print(len(x_test))

    x_train = x_train.reshape(4001, 1, 150)
    print(x_train.shape)
    x_test = x_test.reshape(1001, 1, 150)
    print(x_test.shape)

    print(y_train.shape)
    print(y_test.shape)


    print(x_train)
    print(y_train)
    print('Build model...')
    model = Sequential()
    model.add(LSTM(300, input_shape=(1, 150)))
    model.add(Dropout(0.2))
    model.add(Dense(units=len(class_names)))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)

    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print("Loss: %.2f" % (loss))
    print("Validation Accuracy: %.2f" % (acc))

    new_text = "hej där vem är du"
    print("New text:", new_text)
    new_text = [new_text[i:i + 1] for i in range(len(new_text) - 1 + 1)]
    new_text = [ord(new_text[i]) for i in range(len(new_text))]
    new_text = np.pad(new_text, (0, 150 - len(new_text)), constant_values=0)
    new_text = np.asarray(new_text)
    new_text = new_text.reshape(1, 1, 150)
    print(new_text)
    new_prediction = model.predict(new_text)

    print(new_prediction)
    print("Final prediction:", class_names[np.argmax(new_prediction)])

    # # Plot training & validation accuracy values
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.axhline(y=0.9, color='r', linestyle='--')
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # plt.show()

if __name__ == "__main__":
    main()

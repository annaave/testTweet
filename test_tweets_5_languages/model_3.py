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
    print(d)
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

    # lowercase all letters
    words = line.split()
    words = [word.lower() for word in words]
    line = ' '.join(words)

    return line

def split_data(frame):
    train, validation = train_test_split(frame, test_size=0.2)
    validation, test = train_test_split(validation, test_size=0.5)
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, validation, test

def encode_ngram_text(text, n):
    text = [text[i:i + n] for i in range(len(text) - n + 1)] #splitting up the text as n-grams
    text = [ord(text[i]) for i in range(len(text))] #converting every character to its unicode
    return text

def pad(data, length):
    new_data = []
    for row in data:
        new_row = np.pad(row, (0, length - len(row)), constant_values=0)
        new_data.append(new_row)
    data = new_data
    return data

def decode(text):
    text = [chr(text[i]) for i in range(len(text))] # converting every character back from unicode to its original
    return text

def main():
    class_names = ["English", "Swedish", "Spanish", "Portuguese", "Russian"]

    all_data = read_all(class_names)

    train, validation, test = split_data(all_data)

    print("Printing first 50 training tweets and its labels before making changes of input representation:")
    print(train[:50])

    test_lines = test[:3]

    x_train = np.asarray([np.asarray(text) for text in train['tweets']])

    y_train = np.asarray([np.asarray(label) for label in train['language']])

    x_validation = np.asarray([np.asarray(text) for text in validation['tweets']])
    y_validation = np.asarray([np.asarray(label) for label in validation['language']])

    x_test = np.asarray([np.asarray(text) for text in test['tweets']])
    y_test = np.asarray([np.asarray(label) for label in test['language']])

    x_train = [encode_ngram_text(line, 1) for line in x_train]
    x_train = pad(x_train, max_length)
    x_validation = [encode_ngram_text(line, 1) for line in x_validation]
    x_validation = pad(x_validation, max_length)
    x_test = [encode_ngram_text(line, 1) for line in x_test]
    x_test = pad(x_test, max_length)
    x_train = np.asarray(x_train)
    x_validation = np.asarray(x_validation)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_validation = np.asarray(y_validation)
    y_test = np.asarray(y_test)

    histo_train = np.histogram(y_train, bins=5)
    histo_vali = np.histogram(y_validation, bins=5)
    histo_test = np.histogram(y_test, bins=5)

    print("Training histogram", histo_train)
    print("Validation histogram", histo_vali)
    print("Test histogram", histo_test)

    y_train = np_utils.to_categorical(y_train, num_classes=5)
    y_validation = np_utils.to_categorical(y_validation, num_classes=5)
    y_test = np_utils.to_categorical(y_test, num_classes=5)

    print("Length of x_train:", len(x_train))
    print("Length of x_validation:", len(x_validation))
    print("Length of x_test:", len(x_test))

    x_train = x_train.reshape(4001, 1, 150)
    print("Shape of x_train:", x_train.shape)
    x_validation = x_validation.reshape(500, 1, 150)
    print("Shape of x_validate:", x_validation.shape)
    x_test = x_test.reshape(501, 1, 150)
    print("Shape of x_test:", x_test.shape)

    print("Shape of y_train:", y_train.shape)
    print("Shape of y_validate:", y_validation.shape)
    print("Shape of y_test:", y_test.shape)

    print("Printing 3 first training tweets and its labels as final input form:")
    print(x_train[:3])
    print(y_train[:3])

    print('Build model...')
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, 150)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(class_names)))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    history = model.fit(x_train, y_train, batch_size=10, epochs=50, validation_data=(x_validation, y_validation), verbose=1)

    loss, acc = model.evaluate(x_validation, y_validation, verbose=1)
    print("Loss: %.2f" % (loss))
    print("Validation Accuracy: %.2f" % (acc))
    #print("Test (hold out data set) Accuracy: %.2f" % (acc))

    loss2, acc3 = model.evaluate(x_test, y_test, verbose=1)
    print("Loss: %.2f" % (loss2))
    print("Test (hold-out-dataset) Accuracy: %.2f" % (acc3))

    # print('\n# Generate predictions for 3 samples')
    # print(test_lines)
    # predictions = model.predict(x_test[:3])
    # print('prediction 1:%.2f' % predictions[0], "Correct label:", y_test[0])
    # print('prediction 2:%.2f' % predictions[1], "Correct label:", y_test[1])
    # print('prediction 3:%.2f' % predictions[2], "Correct label:", y_test[2])


    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    #plt.axhline(y=0.85, color='r', linestyle='--')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('model_3_test_fig.png')

if __name__ == "__main__":
    main()

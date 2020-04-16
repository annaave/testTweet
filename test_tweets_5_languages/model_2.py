#https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/
#https://stackoverflow.com/questions/18658106/quick-implementation-of-character-n-grams-for-word
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv2D
from keras.layers import Dropout, Flatten # Different types of layers
import numpy as np
import sys
#import matplotlib.pyplot as plt
import re
np.set_printoptions(threshold=sys.maxsize)

vocab_size = 2000
embedding_dim = 64
max_length = 150
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

def add_the_language(file, language):
    one_lang = pd.read_csv(file, sep='\t', header=None)
    one_lang = one_lang.drop(index=0)
    one_lang.columns = ["tweets"]
    one_lang["language"] = language
    one_lang = one_lang.reset_index(drop=True)

    # Remove url:s
    for i in range(len(one_lang)):
        one_lang['tweets'][i] = clean_up(one_lang['tweets'][i])

    return one_lang

def add_other_languages(file1, file2, file3, file4):
    other_language1 = pd.read_csv(file1, sep='\t', header=None)
    other_language1 = other_language1.drop(index=0)
    other_language2 = pd.read_csv(file2, sep='\t', header=None)
    other_language2 = other_language2.drop(index=0)
    other_language3 = pd.read_csv(file3, sep='\t', header=None)
    other_language3 = other_language3.drop(index=0)
    other_language4 = pd.read_csv(file4, sep='\t', header=None)
    other_language4 = other_language4.drop(index=0)
    other_languages = pd.concat([other_language1, other_language2, other_language3, other_language4], ignore_index=True, sort=False)
    other_languages.columns = ["tweets"]
    other_languages["language"] = "Other"

    # Remove url:s
    for i in range(len(other_languages)):
        other_languages['tweets'][i] = clean_up(other_languages['tweets'][i])

    return other_languages

def clean_up(line):
    # remove url
    line = re.sub(r'http\S+', '', line)
    return line

def n_gram_text(text, n):
    #print(text)
    #print("Length of line:", len(text))
    text = [text[i:i + n] for i in range(len(text) - n + 1)]
    #print("Length of line:", len(text))
    #print(text)
    return text

def text_to_unicode(data):
    new_data = []
    for row in data:
        new_row = []
        for part in row:
            #part = part.encode("utf-8")
            part = ord(part)
            new_row.append(part)
        row = new_row
        new_data.append(new_row)
    return new_data

def pad(a, length):
    #a = np.pad(a, (0, length-len(a)), constant_values = 0)
    arr = np.zeros(length)
    arr[:len(a)] = a
    return arr
    #return a

def lstm_model(train_padded, train_label_padd, validation_padded, vali_label_pad):
    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000,
        # and output embedding dimension of size 64 we set at the top
        tf.keras.layers.Embedding(max_feature, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # Add a Dense layer with 2 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_epochs = 5
    history = model.fit(train_padded, train_label_padd, epochs=num_epochs,
                        validation_data=(validation_padded, vali_label_pad), verbose=1)

    loss, acc = model.evaluate(validation_padded, vali_label_pad, verbose=1)
    print("Loss: %.2f" % (loss))
    print("Validation Accuracy: %.2f" % (acc))


def main():
    the_language = add_the_language("1000_Eng_tweets.csv", "English")
    other_languages = add_other_languages("1000_Rus_tweets.csv", "1000_Swe_tweets.csv", "1000_Spa_tweets.csv", "1000_Por_tweets.csv" )
    pd.options.display.max_colwidth = 500

    all_data = pd.concat([the_language, other_languages], ignore_index=True, sort=False)

    all_data_arr = all_data.to_numpy()
    tweets = []
    labels = []

    np.random.shuffle(all_data_arr)
    for row in all_data_arr:
        tweets.append(row[0])
        labels.append(row[1])


    #Printing all sizes of the data, training data and validation data.
    print("Number of tweets:", len(tweets))
    print("Number of labels:", len(labels))
    print()
    print("Text before converting to n-grams:")
    print(tweets[0])
    print(tweets[1])
    print(tweets[2])

    tweets = [n_gram_text(line, 1) for line in tweets]
    print()
    print("Text after converting to n-grams:")
    print(tweets[0])
    print(tweets[1])
    print(tweets[2])

    train_size = int(len(tweets) * training_portion)

    train_tweets = tweets[0: train_size]
    train_labels = labels[0: train_size]

    validation_tweets = tweets[train_size:]
    validation_labels = labels[train_size:]

    print("Labels:", set(labels))
    print()
    print("Training size:", train_size)
    print("Number of training tweets:", len(train_tweets))
    print("Number of training labels:", len(train_labels))
    print("Number of validation tweets:", len(validation_tweets))
    print("Number of validation labels:", len(validation_labels))


    print()
    print("No padding and before encoded as unicode, of first 3 tweets in training data:")
    print(tweets[0])
    print(tweets[1])
    print(tweets[2])
    print(labels[0])
    print(labels[1])
    print(labels[2])

    train_tweets = text_to_unicode(train_tweets)
    validation_tweets = text_to_unicode(validation_tweets)

    print(train_tweets[:3])

    train_padded = []
    validation_padded = []
    for i in range(len(train_tweets)):
        padding = pad(train_tweets[i], max_length)
        train_padded.append(padding)
    print()
    print("Length of 1st tweet in training data without padding:", len(train_tweets[0]))
    print("Length of 1st tweet in training data with padding:", len(train_padded[0]))


    for i in range(len(validation_tweets)):
        validation_padded.append(pad(validation_tweets[i], max_length))
    print("Length of 1st tweet in validation data without padding:", len(validation_tweets[0]))
    print("Length of 1st tweet in validation data with padding:", len(validation_padded[0]))

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    print()
    print("With both padding and encoding as unicode, of first 3 tweets in training data:")
    print(train_padded[0])
    print(train_padded[1])
    print(train_padded[2])
    print(training_label_seq[0])
    print(training_label_seq[1])
    print(training_label_seq[2])

    x_train = np.asarray(train_padded)
    y_train = np.asarray(training_label_seq)
    x_test = np.asarray(validation_padded)
    y_test = np.asarray(validation_label_seq)

# Calculating average and maximum values of the original length of the 1-gram input data string
    med = 0
    max_len = 0
    for row in tweets:
        med += len(row)
        if len(row) > max_len:
            max_len = len(row)

    print("Average length of input data:", med/(len(tweets)-1))
    print("Maximum length of input data:", max_len)

    print(x_train[:3])
    print(len(y_train))
    print(len(x_test))
    print(len(y_test))

    #train_padded = np.asanyarray(train_padded)
    #train_padded = train_padded.reshape(1, 4001, 150)
    #training_label_seq = np.asanyarray(training_label_seq)
    #validation_padded = np.asanyarray(validation_padded)
    #validation_padded = validation_padded.reshape(1, 1001, 150)
    #validation_label_seq = np.asanyarray(validation_label_seq)
    x_train = x_train.reshape(1, 4001, 150)
    x_test = x_test.reshape(1, 1001, 150)
    print(x_train.shape)
    print(x_test.shape)
    input_length = x_train.shape[1]
    input_dim = x_train.shape[2]
    output_dim = 2
    print(output_dim)

    model = Sequential()
    # model.add(LSTM(embedding_dim, input_shape=x_train.shape))
    # model.add(Dense(2, activation='softmax'))
    #
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # Build the model
    # model = Sequential()

    # I arbitrarily picked the output dimensions as 4
    model.add(LSTM(4, input_dim=input_dim, input_length=input_length))
    # The max output value is > 1 so relu is used as final activation.
    model.add(Dense(output_dim, activation='relu'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    # num_epochs = 5
    # model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=1)
    # Set batch_size to 7 to show that it doesn't have to be a factor or multiple of your sample size
    history = model.fit(x_train, y_train, batch_size=7, epochs=3, verbose=1)

    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    #loss, acc = model.evaluate(validation_padded, validation_label_seq, verbose=1)
    print("Loss: %.2f" % (loss))
    print("Validation Accuracy: %.2f" % (acc))


    # pritning all data to see that they are correct
    # for line in tweets:
    #     print(line)
    #
    # for line in labels:
    #     print(line)
    #print(the_language.to_string())
    #print(other_languages.to_string())


    # new_text = "hej jag ääär Coola Killen"
    # new_text = n_gram_text(new_text, 1)
    # print(new_text)
    # new_text = [part.encode("utf-8") for part in new_text]
    # print(new_text)

if __name__ == "__main__":
    main()

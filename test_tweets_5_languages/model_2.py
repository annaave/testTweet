#https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/
#https://stackoverflow.com/questions/18658106/quick-implementation-of-character-n-grams-for-word
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
#import matplotlib.pyplot as plt
import re
np.set_printoptions(threshold=sys.maxsize)

max_feature = 10000
embedding_dim = 32
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
    for row in data:
        part = [part.encode("utf-8") for part in row]

#def text_to_sequence(data):

#def data_padding(data):

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

    print("Text before and after converting to n-grams:")
    print(tweets[0])
    print(tweets[1])
    print(tweets[2])

    tweets = [n_gram_text(line, 2) for line in tweets]

    print(tweets[0])
    print(tweets[1])
    print(tweets[2])

    train_size = int(len(tweets) * training_portion)

    train_tweets = tweets[0: train_size]
    train_labels = labels[0: train_size]

    validation_tweets = tweets[train_size:]
    validation_labels = labels[train_size:]

    print("Training size:", train_size)
    print("Number of training tweets:", len(train_tweets))
    print("Number of training labels:", len(train_labels))
    print("Number of validation tweets:", len(validation_tweets))
    print("Number of validation labels:", len(validation_labels))

    print(tweets[0])
    print(tweets[1])
    print(tweets[2])
    print(labels[0])
    print(labels[1])
    print(labels[2])


    # pritning all data to see that they are correct
    # for line in tweets:
    #     print(line)
    #
    # for line in labels:
    #     print(line)
    #print(the_language.to_string())
    #print(other_languages.to_string())

    tokenizer = Tokenizer(num_words=max_feature, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_tweets)
    word_index = tokenizer.word_index
    print(dict(list(word_index.items())[0:100]))

    train_sequences = tokenizer.texts_to_sequences(train_tweets)

    #train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    new_text = "hej jag ääär Coola Killen"
    new_text = n_gram_text(new_text, 1)
    print(new_text)
    new_text = [part.encode("utf-8") for part in new_text]
    print(new_text)

if __name__ == "__main__":
    main()

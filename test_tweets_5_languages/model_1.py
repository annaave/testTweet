
# import pandas as pd
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

vocab_size = 5000
embedding_dim = 64
max_length = 15
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

all_data = []
tweets = []
labels = []

with open("1000_Eng_tweets_label.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        all_data.append(row)
with open("1000_Swe_tweets_label.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        all_data.append(row)

np.random.shuffle(all_data)
print(all_data)
print(len(all_data))

for row in all_data:
    tweets.append(row[0])
    labels.append(row[1])


print(len(tweets))
print(len(labels))
print(labels)

train_size = int(len(tweets) * training_portion)

train_tweets = tweets[0: train_size]
train_labels = labels[0: train_size]

validation_tweets = tweets[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_tweets))
print(len(train_labels))
print(len(validation_tweets))
print(len(validation_labels))

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_tweets)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

train_sequences = tokenizer.texts_to_sequences(train_tweets)
print(train_sequences[10])
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))
print(train_sequences[10])
print(train_padded[10])


validation_sequences = tokenizer.texts_to_sequences(validation_tweets)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)

print(set(labels))
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))


print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_tweet(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_tweet(train_padded[10]))
print('---')
print(train_tweets[10])


model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000,
    # and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 2 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 5
history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq), verbose=1)



loss, acc = model.evaluate(validation_padded, validation_label_seq, verbose=1)
print("Loss: %.2f" % (loss))
print("Validation Accuracy: %.2f" % (acc) )


y_pred = model.predict_classes(validation_padded)
print(tf.math.confusion_matrix(labels=validation_label_seq, predictions=y_pred))


new_test = ["hej på dig då!"]
tokenizer.fit_on_texts(new_test)
new_sequences = tokenizer.texts_to_sequences(new_test)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

new_prediction = model.predict(new_padded)
#labels = ['English', 'Swedish']
#print(new_prediction, labels[np.argmax(new_prediction)])

new_labels = ['none', 'English', 'Swedish']

print("New test:", new_test)
print(new_prediction)
print("Final prediction:", new_labels[np.argmax(new_prediction)])


new_test_2 = ["well brother"]
tokenizer.fit_on_texts(new_test_2)
new_sequences_2 = tokenizer.texts_to_sequences(new_test_2)
new_padded_2 = pad_sequences(new_sequences_2, maxlen=max_length, padding=padding_type, truncating=trunc_type)

new_prediction_2 = model.predict(new_padded_2)

print("New test 2:", new_test_2)
print(new_prediction_2)
print("Final prediction 2:", new_labels[np.argmax(new_prediction_2)])


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    # plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

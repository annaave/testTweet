import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

vocab_size = 500
embedding_dim = 64
max_length = 150
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'


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

    # remove emojis
    line = remove_emojies(line)

    # remove excessive signs
    # remove_characters = re.compile('[/(){}\[\]\|.,;:!?"<>^*&%$]')
    # line = remove_characters.sub('', line)
    return line


def remove_emojies(text):
    # Ref: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1#gistcomment-3208085
    # Ref: https://en.wikipedia.org/wiki/Unicode_block
    emojies = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "])")
    text = re.sub(emojies, r'', text)
    return text


def split_data(frame):
    train, validation = train_test_split(frame, test_size=0.2)
    validation, test = train_test_split(validation, test_size=0.5)
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, validation, test


def encode_ngram_text(text, n):
    text = [text[i:i + n] for i in range(len(text) - n + 1)]  # splitting up the text as n-grams
    text = [ord(text[i]) for i in range(len(text))]  # converting every character to its unicode
    return text


def pad(data, length):
    new_data = []
    for row in data:
        new_row = np.pad(row, (0, length - len(row)), constant_values=0)
        new_data.append(new_row)
    data = new_data
    return data


def decode(text):
    text = [chr(text[i]) for i in range(len(text))]  # converting every character back from unicode to its original
    return text


def main():
    class_names = ["English", "Swedish", "Spanish", "Portuguese", "Russian"]

    all_data = read_all(class_names)

    train, validation, test = split_data(all_data)

    x_train = np.asarray([np.asarray(text) for text in train['tweets']])

    y_train = np.asarray([np.asarray(label) for label in train['language']])

    x_validation = np.asarray([np.asarray(text) for text in validation['tweets']])
    y_validation = np.asarray([np.asarray(label) for label in validation['language']])

    x_test = np.asarray([np.asarray(text) for text in test['tweets']])
    y_test = np.asarray([np.asarray(label) for label in test['language']])

    print(x_train[0])
    print(x_train[1])
    # print(all_data['tweets'])
    # print(all_data['language'])

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, char_level=True)
    tokenizer.fit_on_texts(all_data['tweets'])
    word_index = tokenizer.word_index
    print(dict(list(word_index.items())[0:10]))

    train_sequences = tokenizer.texts_to_sequences(x_train)
    x_train_pad = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    validation_sequences = tokenizer.texts_to_sequences(x_validation)
    x_validation_pad = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type,
                                     truncating=trunc_type)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    x_test_pad = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    print(x_train_pad[0])
    print(y_train[0])

    print(x_train_pad[1])
    print(y_train[1])

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dense(embedding_dim, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    num_epochs = 15
    history = model.fit(x_train_pad, y_train, batch_size=64, epochs=num_epochs,
                        validation_data=(x_validation_pad, y_validation), verbose=1)

    loss, acc = model.evaluate(x_validation_pad, y_validation, verbose=1)
    print("Loss: %.2f" % loss)
    print("Validation Accuracy: %.2f" % acc)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('model_5_lang_test_fig.png')

    y_pred = model.predict_classes(x_test_pad)
    print(tf.math.confusion_matrix(labels=y_test, predictions=y_pred))

    print('\n# Generate predictions for 6 samples from the hold-out dataset (testing set)')
    predictions = model.predict(x_test_pad)
    print('prediction 1:', x_test[0], predictions[0], "Correct label:", y_test[0])
    print('prediction 2:', x_test[1], predictions[1], "Correct label:", y_test[1])
    print('prediction 3:', x_test[2], predictions[2], "Correct label:", y_test[2])
    print('prediction 4:', x_test[3], predictions[3], "Correct label:", y_test[3])
    print('prediction 5:', x_test[4], predictions[4], "Correct label:", y_test[4])
    print('prediction 6:', x_test[5], predictions[5], "Correct label:", y_test[5])


if __name__ == "__main__":
    main()

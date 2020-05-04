import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean(data):
    # Clean tweet data
    for i in range(len(data)):
        data['tweets'][i] = clean_up(data['tweets'][i])


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
    remove_characters = re.compile('[/(){}\[\]\|.,;:!?"<>^*&%$]')
    line = remove_characters.sub('', line)
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
    train, validation = train_test_split(frame, test_size=0.20)
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


# returns maximum integer value in the tokenized dictionary of a character in the dataset
def max_dict_value(data):
    n = 0
    for row in data:
        for element in row:
            if element > n:
                n = element
    return n


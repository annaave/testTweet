import pandas as pd
import numpy as np


def add_label_csv(file, lang, newFile):
    data = pd.read_csv(file, sep='\t', header=None)
    data.columns = ["tweets"]
    data["language"] = lang
    data.to_csv(newFile, index=False)


def toDataframe(file):
    data = pd.read_csv(file, na_values='?', dtype={'ID': str}).dropna().reset_index()
    return data


def cleanTweets(data, lang, newFile):
    # print(data['tweets'].iloc[[1]].str.split())
    # data.set_option("display.max_rows", None, "display.max_columns", None)
    # print((data['tweets']))
    pd.options.display.max_colwidth = 500

    clean_data = pd.DataFrame(columns=['tweets', 'language'])

    for i in range(len(data)):
        text = data['tweets'][i]
        text = cleanTweet_line(text)
        new_row = {'tweets': text, 'language': lang}
        clean_data = clean_data.append(new_row, ignore_index=True)

    clean_data = clean_data.drop(index=0)
    clean_data.to_csv(newFile, index=False)
    print(clean_data)
    return clean_data


def cleanTweet_line(line):
    #stop_words = set(stopwords.words(language))
    #word_tokens = word_tokenize(line)
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #print(word_tokens)

    #remove emojis
    text = remove_emojies(line)

    #remove excessive signs
    #https://gist.github.com/susanli2016/0a65cbb9dcd685d96458d106ec8cc499#file-text_preprocessing_lstm-py
    remove_characters = re.compile('[/(){}\[\]\|.,;:!?"<>^*&%$]')
    text = remove_characters.sub('', text)

    #remove url
    text = re.sub(r'http\S+', '', text)

    words = text.split()
    words = [word.lower() for word in words]
    #print(words, "number of words:", len(words))

    for word in words:
        if word[0] == '@':
            words.remove(word)
        elif len(word) <= 1:
            words.remove(word)
        elif word[0] == '#':
            words.remove(word)
        elif word == 'rt':
            words.remove(word)
    for word in words:
        if word[0] == '@':
            words.remove(word)
        if word[0] == '#':
            words.remove(word)
    #print(words, "number of words:", len(words))

    text = ' '.join(words)
    return text


def remove_emojies(text):
  # Ref: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1#gistcomment-3208085
  # Ref: https://en.wikipedia.org/wiki/Unicode_block
  EMOJI_PATTERN = re.compile(
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
  text = re.sub(EMOJI_PATTERN, r'', text)
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

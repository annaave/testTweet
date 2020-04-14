# link to text about n-gram model in python
# https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/
# https://stackoverflow.com/questions/18658106/quick-implementation-of-character-n-grams-for-word

import pandas as pd
import csv
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re


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
    print(text)
    print("Length of line:", len(text))
    text = [text[i:i + n] for i in range(len(text) - n + 1)]
    print("Length of line:", len(text))
    return text

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

    #test for the n-gram function
    # line = "Hej jag heter Anna!"
    # new_line = n_gram_text(line, 2)
    # print(new_line)


    # oritning all data to see that they are correct
    # for line in tweets:
    #     print(line)
    #
    # for line in labels:
    #     print(line)
    #print(the_language.to_string())
    #print(other_languages.to_string())

if __name__ == "__main__":
    main()

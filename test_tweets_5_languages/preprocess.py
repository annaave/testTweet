import pandas as pd
from nltk.tokenize import word_tokenize


def add_label_csv(file, lang, newFile):
    data = pd.read_csv(file, sep='\t', header=None)
    data.columns = ["tweets"]
    data["language"] = lang
    data.to_csv(newFile, index=False)

def toDataframe(file):
    data = pd.read_csv(file, na_values='?', dtype={'ID': str}).dropna().reset_index()
    return data

def cleanTweet(data):
        print(word_tokenize(data['tweets'].iloc[[1]]))


def main():
    #add_label_csv("new_Eng_tweets.csv", "English", "Eng_tweets.csv")
    eng_data = toDataframe("Eng_tweets.csv")
    cleanTweet(eng_data)

if __name__ == "__main__":
    main()


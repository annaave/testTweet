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
        #print(data['tweets'].iloc[[1]].str.split())
        #data.set_option("display.max_rows", None, "display.max_columns", None)
        #print((data['tweets']))

        text = data['tweets'].iloc[[1]]

        # split into words by white space
        words = text[1].split()
        print(len(words))
        # convert to lower case
        words = [word.lower() for word in words]


        for i in range(len(words)):
            print(words[i])


def main():
    #add_label_csv("new_Eng_tweets.csv", "English", "Eng_tweets.csv")
    eng_data = toDataframe("Eng_tweets.csv")
    #print(eng_data)
    cleanTweet(eng_data)

if __name__ == "__main__":
    main()


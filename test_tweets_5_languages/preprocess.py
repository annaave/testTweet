import pandas as pd

def add_label_csv(file, lang, newFile):
    data = pd.read_csv(file,sep='\t', header=None)
    data.columns = ["tweets"]
    data["language"] = lang
    data.to_csv(newFile, index=False)

    print(data)

def main():
    add_label_csv("new_Eng_tweets.csv", "English", "Eng_tweets.csv")

if __name__ == "__main__":
    main()


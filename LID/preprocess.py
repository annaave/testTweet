import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ReadFiles:
    def __init__(self, files):
        self.files = files
        self.files_df = pd.DataFrame()
        self.labeled_files = pd.DataFrame()
        self.df = pd.DataFrame()
        with open(self.files) as f:
            self.files_df = pd.read_csv(self.files, sep=',', engine='python', usecols=['file', 'language'])
        #self.file_names = [line.rstrip() for line in self.files_df['file']]

    def add_labels(self):
        language = []
        readable_files = []
        for i in range(len(self.files_df)):
            language.append(self.files_df['language'][i])

        for i in range(len(self.files_df)):
            data = pd.read_csv(self.files_df['file'][i], sep='\t', header=None)
            data.columns = ["tweets"]
            data["language"] = language[i]
            data.to_csv("labeled_" + str(self.files_df['language'][i]) + ".csv", index=False)
            string = "labeled_" + str(self.files_df['language'][i]) + ".csv"
            readable_files.append(string)
        frame = pd.DataFrame(readable_files)
        self.labeled_files = frame
        frame.to_csv("data_readable_files.csv", index=False)

    def read_all_files(self):
        max_rows = 5000

        for row in self.labeled_files:
            df_new = (pd.read_csv(row, usecols=['tweets', 'language']).dropna(subset=['tweets', 'language'])
                      .assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
            df_new = df_new.drop(index=0)
            df_new = df_new.reset_index(drop=True)
            self.df = self.df.append(df_new, ignore_index=True)

        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def clean_data(self):
        # Clean tweet data
        for i in range(len(self.df)):
            self.df['tweets'][i] = self.clean_line(self.df['tweets'][i])

    @staticmethod
    def clean_line(line):
        # remove url
        line = re.sub(r'http\S+', '', line)

        # remove emojis
        line = remove_emojies(line)

        return line

    @staticmethod
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

    def save_all_data(self):
        self.df.to_csv('all_data_preprocessed.csv')

    @staticmethod
    def split_data(frame):
        train, validation = train_test_split(frame, test_size=0.20)
        validation, test = train_test_split(validation, test_size=0.5)
        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)
        # test = validation
        return train, validation, test


def main():
    pd.options.display.max_colwidth = 1000

    files = ReadFiles('data_files_LID.csv')
    print(files.files_df)
    files.read_all_files()
    print(files.labeled_files)




if __name__ == "__main__":
    main()

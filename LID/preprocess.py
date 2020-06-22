import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os import path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
Preprocessing class which reads collected text data, adds labels to it and 
convert it to the desired format for further modelling. As for now, the class 
can preprocess data for LSTM models and fastText models.
"""


class Preprocess:
    def __init__(self, files, model_type, class_names, raw_data_path, label_data_path, vocab_size, num_lang,
                 oov_tok='<OOV>', trunc_type='post', padding_type='post', max_length=150):
        self.model_type = model_type
        self.class_names = class_names  # The names of the classes/labels for all the data samples
        self.num_lang = num_lang
        self.d = dict(zip(self.class_names, range(0, self.num_lang)))
        print(self.d)
        try:
            if self.model_type == 'LSTM' or self.model_type == 'fastText':
                self.max_rows = 5000  # Maximum number of rows for each language dataset added to the total dataset
                pd.options.display.max_colwidth = 1000  # Displaying longer lines when printing dataFrames
                self.file_names = pd.DataFrame()
                self.data = pd.DataFrame()
                self.vocab_size = vocab_size
                self.oov_tok = oov_tok
                self.trunc_type = trunc_type
                self.padding_type = padding_type
                self.max_length = max_length  # Maximum length of a preprocessed data sample
                self.raw_data_path = raw_data_path
                self.label_data_path = label_data_path
                with open(files) as f:
                    self.file_names = pd.read_csv(files, sep=',', engine='python', usecols=['file', 'language'])

                if self.model_type == 'fastText':
                    self.fasttext_train_file = 'training_data_fasttext.txt'
            else:
                raise ValueError('Invalid model type')
        except ValueError as exp:
            print('Only LSTM or fastText are valid model_types!')

    # Function to add labels to every data sample and create a
    # file with a list of the names of the new labeled files.
    def add_labels(self):
        if path.exists("data_readable_files.csv"):
            print('Files with labels already exists!')
        else:
            language = []
            readable_files = []
            for i in range(len(self.file_names)):  # Retrieve the language for each corresponding dataset
                language.append(self.file_names['language'][i])

            for i in range(len(self.file_names)):  # Create a new file with the label with every data sample
                path_file = self.raw_data_path + self.file_names['file'][i]
                data = pd.read_csv(str(path_file), sep='\t', header=None)
                data.columns = ["tweets"]
                data["language"] = language[i]
                data.to_csv(self.label_data_path + "labeled_" + str(self.file_names['language'][i]) + ".csv",
                            index=False)
                string = "labeled_" + str(self.file_names['language'][i]) + ".csv"
                readable_files.append(string)
            frame = pd.DataFrame(readable_files)
            frame.columns = ["file"]
            # Save a file with a list of all the dataset files with labeled data samples
            frame.to_csv("data_readable_files.csv", index=False)

    def read_all_files(self):
        if path.exists("data_readable_files.csv"):
            labeled_files = pd.read_csv("data_readable_files.csv")
            for i in range(len(labeled_files)):
                df_new = (
                    pd.read_csv(self.label_data_path + labeled_files["file"][i], usecols=['tweets', 'language']).dropna(
                        subset=['tweets', 'language'])
                    .assign(tweets=lambda x: x.tweets.str.strip()).head(self.max_rows))

                df_new = df_new.drop(index=0)
                self.data = self.data.append(df_new)

            if self.model_type == 'LSTM':
                self.data['language'] = self.data['language'].map(self.d, na_action='ignore')

            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.data.reset_index(drop=True, inplace=True)
            self.data.drop_duplicates(subset="tweets", keep='first', inplace=True)
            self.data.reset_index(drop=True, inplace=True)

        else:
            print("Create files with labels first, with class method add_labels()!")

    def clean_data(self):
        # Clean text of url:s and emojies
        for i in range(len(self.data)):
            row = self.data.loc[i, 'tweets']
            #line = re.sub(r'http\S+', '', row)
            text = re.sub(r'http\S+', '', row)
            #text = remove_emojies(line)
            self.data.loc[i, 'tweets'] = text
            # line = re.sub(r'http\S+', '', self.data['tweets'][i])
            # self.data['tweets'][i] = text
            # self.data['tweets'][i] = line
            #self.data.loc[i, 'tweets'] = line
            # text = remove_emojies(self.data['tweets'][i])

    def split_clean_save_data(self, clean_data):
        if self.model_type == 'LSTM':
            if clean_data:
                self.clean_data()
            train, validation = train_test_split(self.data, test_size=0.20)
            validation, test = train_test_split(validation, test_size=0.5)
            train = train.reset_index(drop=True)
            validation = validation.reset_index(drop=True)
            test = test.reset_index(drop=True)
            train = pd.DataFrame(train)
            validation = pd.DataFrame(validation)
            test = pd.DataFrame(test)
            train.to_csv('training_data_9.csv', index=False)
            validation.to_csv('validation_data_9.csv', index=False)
            test.to_csv('test_data_9.csv', index=False)
        if self.model_type == 'fastText':
            if clean_data:
                self.clean_data()
            train, test = train_test_split(self.data, test_size=0.10)
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
            train = pd.DataFrame(train)
            test = pd.DataFrame(test)
            self.create_train_file_fasttext(train)
            test.to_csv('test_data_fasttext.txt', index=False)

    def create_train_file_fasttext(self, train_data):
        """ Creates a text file such that for each tri-gram: <__label__, trigram>
                where label is the language. FastText takes a file as input for training.
                Returns: File name of the created file.
            """
        train_file = open(self.fasttext_train_file, "w+")
        for i in range(len(train_data)):
            label = "__label__" + train_data["language"].iloc[i]
            text = " ".join(train_data["tweets"].iloc[i])
            train_file.write(label + " " + text + "\n")
        train_file.close()

    def tokenize_train(self, file_path, char_level):
        train_data = pd.read_csv(file_path)
        x_train = np.asarray([np.asarray(text) for text in train_data['tweets']])
        y_train = np.asarray([np.asarray(label) for label in train_data['language']])
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_tok, char_level=char_level)
        tokenizer.fit_on_texts(train_data['tweets'])
        word_index = tokenizer.word_index
        print('Created dictionary from tokenizer of training data, here are the top 10 types:')
        print(dict(list(word_index.items())[0:10]))
        train_sequences = tokenizer.texts_to_sequences(x_train)
        x_train_pad = pad_sequences(train_sequences, maxlen=self.max_length, padding=self.padding_type,
                                    truncating=self.trunc_type)
        train_object = {
            'x_train': x_train,
            'y_train': y_train,
            'x_train_pad': x_train_pad
        }
        return tokenizer, train_object

    def tokenize(self, file_path, tokenizer):
        current_data = pd.read_csv(file_path)
        x_data = np.asarray([np.asarray(text) for text in current_data['tweets']])
        y_data = np.asarray([np.asarray(label) for label in current_data['language']])
        data_sequences = tokenizer.texts_to_sequences(x_data)
        x_data_pad = pad_sequences(data_sequences, maxlen=self.max_length, padding=self.padding_type,
                                   truncating=self.trunc_type)
        data_object = {
            'x_data': x_data,
            'y_data': y_data,
            'x_data_pad': x_data_pad
        }
        return data_object

    def tokenize_line(self, line, tokenizer):
        text = [line]
        print(text)
        data_sequence = tokenizer.texts_to_sequences(text)
        data_padded = pad_sequences(data_sequence, maxlen=self.max_length, padding=self.padding_type,
                                    truncating=self.trunc_type)
        return data_padded


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

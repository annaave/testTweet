import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional
import fasttext
from sklearn.metrics import confusion_matrix, classification_report


# Class to create and train a machine learning model
class TrainModel:
    def __init__(self, model_type, vocab_size, embedding_dim, class_names, bat_size, num_epochs):
        if model_type == 'lstm':
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.class_names = class_names
            self.bat_size = bat_size
            self.num_epochs = num_epochs
            self.model = Sequential()
            self.model.add(Embedding(vocab_size, embedding_dim))
            self.model.add(Bidirectional(LSTM(embedding_dim)))
            self.model.add(Dense(embedding_dim, activation='relu'))
            self.model.add(Dense(len(class_names), activation='softmax'))
            self.model.summary()
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    def train_model(self, x_train_pad, y_train, x_validation_pad, y_validation, history_path):
        history = self.model.fit(x_train_pad, y_train, batch_size=self.bat_size, epochs=self.num_epochs,
                                 validation_data=(x_validation_pad, y_validation), verbose=1)
        np.save(history_path, history.history)

    def save_model(self, name):
        file_path = '/home/myuser/testTweet/LID/saved_model/' + name
        self.model.save(file_path)


class FastText:
    def __init__(self, file_name, train_file, labels):
        self.file_name = file_name
        train_data = pd.read_csv(train_file)
        train_data = [labels[row['language']] for row in train_data]
        self.train_data = train_data

    def create_train_file(self):
        """ Creates a text file such that for each tri-gram: <__label__, trigram>
                where label is the language. FastText takes a file as input for training.
                Returns: File name of the created file.
            """
        train_file = open(self.file_name, "w+")
        for i in range(len(self.train_data)):
            label = "__label__" + self.train_data["language"].iloc[i]
            text = " ".join(self.train_data["tweets"].iloc[i])
            train_file.write(label + " " + text + "\n")

        train_file.close()
        return self.file_name

    def train_fast_text_model(self):
        model = fasttext.train_supervised(self.file_name, wordNgrams=0, min_count=1, minn=1, lr=0.5, epoch=30,
                                          ws=3, label_prefix='__label__', dim=50)
        print(model)
        return model

    def get_test_pred(self, test_set, model):
        """
        Input: TestSet : <Language, WordTrigrams> Pairs
        Ouput: List of <ActualLabel, PredictedLabel>
        """
        y_actual, y_pred = [], []
        for i in range(len(test_set)):
            y_actual.append("__label__" + test_set["language"].iloc[i])
            pred = model.predict([" ".join(test_set["tweets"].iloc[i])])[0][0]
            y_pred.append(pred)
        return [y_actual, y_pred]

def main():
    vocab_size = 300
    embedding_dim = 128
    num_epochs = 15
    bat_size = 32
    max_length = 150
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'

    class_names = ["English", "Swedish", "Spanish", "Portuguese", "Russian"]


    #----- LSTM model -----
    lstm_model = TrainModel()
    lstm_model.create_lstm_model(vocab_size, embedding_dim, class_names)
    x_train, x_train_pad, y_train, x_validation, x_validation_pad, y_validation, x_test, x_test_pad, y_test =\
        lstm_model.tokenize(vocab_size, oov_tok, trunc_type, padding_type, max_length)

    # history_path = '/home/myuser/testTweet/LID/saved_model/history_not_preprocessed.npy'
    # lstm_model.train_model(bat_size, num_epochs, x_train_pad, y_train, x_validation_pad, y_validation, history_path)
    # lstm_model.save_model("lstm_model_not_preprocessed")

    # ----- fastText model -----
    fast_text = FastText('fastText_training_data.txt', '/home/myuser/testTweet/LID/data/2000/training_data_not_cleaned.csv', class_names)
    fast_text.create_train_file()
    fast_text__model = fast_text.train_fast_text_model()
    y_actual, y_pred = fast_text.get_test_pred(y_test, fast_text__model)
    print(confusion_matrix(y_actual, y_pred))
    print(classification_report(y_actual, y_pred))


if __name__ == "__main__":
    main()
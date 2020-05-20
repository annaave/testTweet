import fasttext
import pandas as pd
import time


class FastText:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = pd.read_csv(test_data)
        self.test_data = self.test_data.replace('\n', ' ', regex=True)

    def train_fast_text_model(self):
        model = fasttext.train_supervised(input=self.train_data, wordNgrams=3, lr=0.5, epoch=15, ws=1,
                                          label_prefix='__label__', dim=300)
        return model

    def get_test_pred(self, model):
        """
        Input: TestSet : <Language, WordTrigrams> Pairs
        Ouput: List of <ActualLabel, PredictedLabel>
        """
        y_actual, y_pred = [], []
        for i in range(len(self.test_data)):
            y_actual.append("__label__" + self.test_data["language"].iloc[i])
            pred = model.predict([" ".join(self.test_data["tweets"].iloc[i])])[0][0]
            y_pred.append(pred)
        return [y_actual, y_pred]

    def speed_test(self, model):
        pred_tot = []
        start = time.time()
        for i in range(4):
            for j in range(len(self.test_data)):
                pred = model.predict([" ".join(self.test_data["tweets"].iloc[j])])[0][0]
                pred_tot.append(pred)
        print(len(pred_tot))
        elapsed_time_fl = (time.time() - start)
        return elapsed_time_fl

import fasttext


class FastText:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def train_fast_text_model(self):
        model = fasttext.train_supervised(input=self.train_data, wordNgrams=0, minn=1, lr=0.5, epoch=15, ws=1,
                                          label_prefix='__label__', dim=50)
        print(model)
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

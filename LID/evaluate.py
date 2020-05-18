import tensorflow as tf
import pandas as pd
import LID.train_model as tm
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix


class EvaluateModel:
    def __init__(self, model):
        self.model = model

    def evaluate_model(self, x_test_pad, y_test, x_validation_pad, y_validation):
        new_model = tf.keras.models.load_model(self.model)

        # Check its architecture
        new_model.summary()

        # Evaluate the restored model
        loss, acc = new_model.evaluate(x_validation_pad, y_validation, verbose=2)
        print("Loss: %.2f" % loss)
        print('Restored model, validation accuracy: {:5.2f}%'.format(100 * acc))
        # Evaluate the restored model
        loss2, acc2 = new_model.evaluate(x_test_pad, y_test, verbose=2)
        print("Loss: %.2f" % loss2)
        print('Restored model, test accuracy: {:5.2f}%'.format(100 * acc2))

        print(new_model.predict(x_test_pad).shape)

    def predict_data(self, x_test_padded):
        new_model = tf.keras.models.load_model(self.model)
        new_y_pred = new_model.predict_classes(x_test_padded)
        return new_y_pred

def save_confusion_matrix(y_test, y_pred, labels):
    y_test_confusion = [labels[row] for row in y_test]
    y_pred_confusion = [labels[row] for row in y_pred]
    data_cn = {'y_Actual': y_test_confusion, 'y_Predicted': y_pred_confusion}
    df = pd.DataFrame(data_cn, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix_2 = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['True labels'],
                                         colnames=['Predicted labels'])

    plt.title("Confusion matrix over test data")
    sn.heatmap(confusion_matrix_2, annot=True, fmt='g', xticklabels=True, yticklabels=True, cmap='Blues')
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.savefig('confusion_matrix_May_06.png')
    plt.close()


def create_confusion(y_test, y_prediction):
    cnf_matrix = confusion_matrix(y_test, y_prediction)
    print(cnf_matrix)
    print(classification_report(y_test, y_prediction))


def save_confusion(y_test, y_prediction, labels):
    y_test_cm = [labels[row] for row in y_test]
    y_prediction_cm = [labels[row] for row in y_prediction]
    data_cn = {'y_Actual': y_test_cm, 'y_Predicted': y_prediction_cm}
    df = pd.DataFrame(data_cn, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix_2 = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['True labels'],
                                     colnames=['Predicted labels'])

    plt.title("Confusion matrix over test data")
    sn.heatmap(confusion_matrix_2, annot=True, fmt='g', xticklabels=True, yticklabels=True, cmap='Blues')
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.savefig('/home/myuser/testTweet/LID/figures/confusion_matrix_May_18.png')
    plt.close()


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
    lstm_model = tm.TrainModel()
    lstm_model.create_lstm_model(vocab_size, embedding_dim, class_names)
    x_train, x_train_pad, y_train, x_validation, x_validation_pad, y_validation, x_test, x_test_pad, y_test = \
        lstm_model.tokenize(vocab_size, oov_tok, trunc_type, padding_type, max_length)

    new_evaluation = EvaluateModel('/home/myuser/testTweet/LID/saved_model/lstm_model_not_preprocessed')
    new_evaluation.evaluate_model(x_test_pad, y_test, x_validation_pad, y_validation)

    y_pred = new_evaluation.predict_data(x_test_pad)
    create_confusion(y_test, y_pred)
    save_confusion(y_test, y_pred, class_names)


if __name__ == "__main__":
        main()

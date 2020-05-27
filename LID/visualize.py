import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pickle


def load_history(file_path):
    history = np.load(file_path, allow_pickle=True).item()
    return history


def plot_graphs(history):
    # Plot training & validation accuracy values
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.title('Model accuracy')
    plt.rc('font', size=14)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('/home/myuser/testTweet/LID/figures/lstm_acc_epochs.png')
    plt.close()

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('/home/myuser/testTweet/LID/figures/lstm_loss_epochs.png')
    plt.close()


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


def char_count(data):
    max_len = 0
    min_len = len(data['tweets'][0])
    count_char = 0
    lengths = []
    for row in data['tweets']:
        if len(row) > max_len:
            max_len = len(row)
        if min_len > len(row):
            min_len = len(row)
        for element in row:
            count_char += 1
        lengths.append(len(row))
    data['tweet length'] = lengths
    return count_char, max_len, min_len


def create_histogram(all_data):
    s = all_data['tweet length']
    ax = s.hist(bins=10, grid=True)
    ax.set_xlabel('Number of characters')
    ax.set_ylabel('Number of tweets')
    ax.set_title('Distribution of tweet lengths')
    fig = ax.get_figure()
    fig.savefig('histogram.png')
    plt.close()

import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import pickle


def load_history(file_path):
    history = np.load(file_path, allow_pickle=True).item()
    return history


def plot_confusion_matrix(y_validation, y_prediction):
    cnf_matrix = confusion_matrix(y_validation, y_prediction)
    print(cnf_matrix)
    print(classification_report(y_validation, y_prediction))


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


def main():
    history_lstm = load_history('/home/myuser/testTweet/LID/saved_model/history.npy')
    plot_graphs(history_lstm)

if __name__  == "__main__":
    main()


from LID.preprocess import Preprocess
from LID.fasttext import FastText
from LID.train_model import TrainModel
from LID.evaluate import EvaluateModel
from LID.visualize import load_history, plot_graphs, plot_bar_chart
import tensorflow as tf
import fasttext


class_names = ['English', 'Swedish', 'Spanish', 'Portuguese', 'Russian']
model_type = 'fastText'
vocab_size = 300
embedding_dim = 128
num_epochs = 15
bat_size = 32

# ------ PREPROCESSING ------
files = '/home/myuser/testTweet/LID/2000_5_data_files_LID.csv'
path_raw_data = '/home/myuser/testTweet/LID/raw_data/2000/'
label_data_path = '/home/myuser/testTweet/LID/data/2000/'
lstm_preprocess = Preprocess(files, model_type=model_type, class_names=class_names, raw_data_path=path_raw_data,
                             label_data_path=label_data_path, vocab_size=vocab_size)
# lstm_preprocess.read_all_files()
# print(lstm_preprocess.data)
# lstm_preprocess.split_clean_save_data(clean_data=False)
# print(lstm_preprocess.data)
# tokenizer, training_data = lstm_preprocess.tokenize_train('/home/myuser/testTweet/LID/training_data.csv', char_level=True)
# validation_data = lstm_preprocess.tokenize('/home/myuser/testTweet/LID/validation_data.csv', tokenizer)
# print(len(validation_data['x_data_pad']))

# ------ LSTM MODEL ------
history_path = '/home/myuser/testTweet/LID/saved_model/history_lstm.npy'
# lstm_model = TrainModel('LSTM', embedding_dim, class_names, bat_size, num_epochs)
# lstm_model.train_model(training_data, validation_data, history_path)
# lstm_model.save_model("lstm_model")

# ----- EVALUATE MODEL ------
# model_path = '/home/myuser/testTweet/LID/saved_model/lstm_model'
# loaded_model = tf.keras.models.load_model(model_path)
#
# test_data = lstm_preprocess.tokenize('/home/myuser/testTweet/LID/test_data.csv', tokenizer)
# lstm_evaluation = EvaluateModel(model_path, validation_data, test_data,)
# lstm_evaluation.evaluate_model()

# ------ VISUALIZE LSTM MODEL ------
# Plot graph of accuracy and loss of model over number of epochs
# history = load_history(history_path)
# plot_graphs(history)

# Bar chart of accuracy over tweet sample lengths
# plot_bar_chart(x_test=test_data['x_data'], x_test_pad=test_data['x_data_pad'], y_test=test_data['y_data'],
#                labels=class_names, model=loaded_model)


# ------ fastText model ------
# fast_text = FastText('training_data_fasttext.txt', '/home/myuser/testTweet/LID/test_data_fasttext.csv')
# fast_text_model = fast_text.train_fast_text_model()
# y_actual, y_pred = fast_text.get_test_pred(fast_text_model)

ft_model = fasttext.train_supervised(input='training_data_fasttext.txt', wordNgrams=0, minn=1, lr=0.5, epoch=15, ws=1, label_prefix='__label__', dim=50)

# print(confusion_matrix(y_actual, y_pred))
# print(classification_report(y_actual, y_pred))


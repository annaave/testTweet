from LID.preprocess import Preprocess
from LID.fastText_2 import FastText
from LID.train_model import TrainModel
from LID.evaluate import EvaluateModel
from LID.visualize import load_history, plot_graphs
from sklearn.metrics import confusion_matrix, classification_report
import timeit


class_names = ['English', 'Swedish', 'Spanish', 'Portuguese', 'Russian']
model_type = 'LSTM'
voc_size = 1000
embedding_dim = 128
num_epochs = 15
bat_size = 32

# ------ PREPROCESSING ------
files = '/home/myuser/testTweet/LID/2000_5_data_files_LID.csv'
path_raw_data = '/home/myuser/testTweet/LID/raw_data/2000/'
label_data_path = '/home/myuser/testTweet/LID/data/2000/'
lstm_preprocess = Preprocess(files, model_type=model_type, class_names=class_names, raw_data_path=path_raw_data,
                             label_data_path=label_data_path, vocab_size=voc_size)
lstm_preprocess.add_labels()
lstm_preprocess.read_all_files()
print(lstm_preprocess.data)
lstm_preprocess.split_clean_save_data(clean_data=False)
print(lstm_preprocess.data)
tokenizer, training_data = lstm_preprocess.tokenize_train('/home/myuser/testTweet/LID/training_data.csv', char_level=True)
validation_data = lstm_preprocess.tokenize('/home/myuser/testTweet/LID/validation_data.csv', tokenizer)

# # ------ PRE-TRAINED MODEL PATH ------
# pre_train_path = '/home/myuser/testTweet/pre_training/saved_model/lstm_model_pre'
#
# # ----- EVALUATE PRE-TRAINED MODEL ------
# test_data = lstm_preprocess.tokenize('/home/myuser/testTweet/LID/4000_test_data.csv', tokenizer)
# lstm_evaluation = EvaluateModel(pre_train_path, validation_data, test_data,)
# lstm_evaluation.evaluate_model()

# ------ LSTM MODEL ------
history_path = '/home/myuser/testTweet/LID/saved_model/history_lstm_2000.npy'
lstm_model = TrainModel('LSTM', embedding_dim, class_names, bat_size, num_epochs, vocab_size=voc_size)
lstm_model.train_model(training_data, validation_data, history_path)
lstm_model.save_model("lstm_model_2000")

# ----- EVALUATE MODEL ------
model_path = '/home/myuser/testTweet/LID/saved_model/lstm_model_2000'

test_data = lstm_preprocess.tokenize('/home/myuser/testTweet/LID/test_data.csv', tokenizer)
lstm_evaluation = EvaluateModel(model_path, validation_data, test_data,)
lstm_evaluation.evaluate_model()


print("Time to predict 4000 tweets for LSTM: ", lstm_evaluation.speed_test(), "seconds")

# Bar chart of accuracy over tweet sample lengths
lstm_evaluation.plot_bar_chart(x_test=test_data['x_data'], x_test_pad=test_data['x_data_pad'],
                               y_test=test_data['y_data'], labels=class_names)

# ------ VISUALIZE LSTM MODEL ------
# Plot graph of accuracy and loss of model over number of epochs
history = load_history(history_path)
plot_graphs(history)



# ------ fastText model ------
# fast_text = FastText('training_data_fasttext.txt', 'test_data_fasttext.txt')
# fast_text_model = fast_text.train_fast_text_model()
# y_actual, y_pred = fast_text.get_test_pred(fast_text_model)
# print("Time to predict 4000 tweets for fastText: ", fast_text.speed_test(fast_text_model), "seconds")
# #
# print(confusion_matrix(y_actual, y_pred))
# print(classification_report(y_actual, y_pred))

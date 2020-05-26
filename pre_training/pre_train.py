import pandas as pd
from LID.preprocess import Preprocess
from LID.evaluate import EvaluateModel
from LID.train_model import TrainModel
from LID.visualize import load_history, plot_graphs

class_names = ['English', 'Swedish', 'Spanish', 'Portuguese', 'Russian']
model_type = 'LSTM'
voc_size = 300
embedding_dim = 128
num_epochs = 30
bat_size = 128

# ------ PREPROCESSING ------
files = '/home/myuser/testTweet/LID/pre_training_files.csv'
path_raw_data = '/home/myuser/testTweet/LID/raw_data'
label_data_path = '/home/myuser/testTweet/LID/data'
lstm_preprocess = Preprocess(files, model_type=model_type, class_names=class_names, raw_data_path=path_raw_data,
                             label_data_path=label_data_path, vocab_size=voc_size)
lstm_preprocess.add_labels()
lstm_preprocess.read_all_files()
print(lstm_preprocess.data)
lstm_preprocess.split_clean_save_data(clean_data=False)
print(lstm_preprocess.data)
tokenizer, training_data = lstm_preprocess.tokenize_train('/home/myuser/testTweet/LID/training_data.csv', char_level=True)
validation_data = lstm_preprocess.tokenize('/home/myuser/testTweet/pre_training/validation_data.csv', tokenizer)


# ------ LSTM MODEL ------
history_path = '/home/myuser/testTweet/pre_training/saved_model/history_lstm_pre.npy'
lstm_model = TrainModel('LSTM', embedding_dim, class_names, bat_size, num_epochs)
lstm_model.train_model(training_data, validation_data, history_path)
lstm_model.save_model("lstm_model_pre")

# ----- EVALUATE MODEL ------
model_path = '/home/myuser/testTweet/pre_training/saved_model/lstm_model_pre'

test_data = lstm_preprocess.tokenize('/home/myuser/testTweet/pre_training/test_data.csv', tokenizer)
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
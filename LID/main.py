from LID.preprocess import Preprocess
from LID.fastText_2 import FastText
from LID.train_model import TrainModel
from LID.evaluate import EvaluateModel
from LID.visualize import load_history, plot_graphs, char_count, create_histogram, create_confusion, save_confusion
from sklearn.metrics import confusion_matrix, classification_report
import re
import pickle
import pandas as pd

class_names = ['Eng', 'Swe', 'Spa', 'Por', 'Rus', 'Ger', 'Pol', 'Ser', 'Cro']
# class_names = ['Eng', 'Swe', 'Spa', 'Por', 'Rus', 'Ger', 'Pol', 'Ser']
# class_names = ['English', 'Swedish', 'Spanish', 'Portuguese', 'Russian', 'German', 'Polish', 'Serbian', 'Croatian']
model_type = 'fastText'
voc_size = 300
embedding_dim = 128
num_epochs = 30
bat_size = 128
number_lang = 9

# ------ PREPROCESSING ------
files = '/home/myuser/testTweet/LID/4000_data_files_LID.csv'
path_raw_data = '/home/myuser/testTweet/LID/raw_data/4000/'
label_data_path = '/home/myuser/testTweet/LID/data/4000/'
lstm_preprocess = Preprocess(files, model_type=model_type, class_names=class_names, raw_data_path=path_raw_data,
                             label_data_path=label_data_path, vocab_size=voc_size, num_lang=number_lang)
# lstm_preprocess.add_labels()
lstm_preprocess.read_all_files()
print(lstm_preprocess.data)
train, test = lstm_preprocess.split_clean_save_data(clean_data=True)
print(lstm_preprocess.data)
#lstm_preprocess.create_train_file_fasttext(train, test)
print(train)
print(test)

# tokenizer, training_data = lstm_preprocess.tokenize_train('/home/myuser/testTweet/LID/training_data.csv', char_level=True)

# saving tokenizer when training a new model
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading tokenizer when using an old model
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# validation_data = lstm_preprocess.tokenize('/home/myuser/testTweet/LID/validation_data.csv', tokenizer)
#
# word_index = tokenizer.word_index
# print('Created dictionary from tokenizer of training data, here are the top 10 types:')
# print(dict(list(word_index.items())[0:10]))

# # ------ PRE-TRAINED MODEL PATH ------
# pre_train_path = '/home/myuser/testTweet/pre_training/saved_model/lstm_model_pre'
#
# # ----- EVALUATE PRE-TRAINED MODEL ------
# test_data = lstm_preprocess.tokenize('/home/myuser/testTweet/LID/4000_test_data.csv', tokenizer)
# lstm_evaluation = EvaluateModel(pre_train_path, validation_data, test_data,)
# lstm_evaluation.evaluate_model()

# ------ LSTM MODEL ------
# history_path = '/home/myuser/testTweet/LID/saved_model/history_lstm_4000.npy'
# lstm_model = TrainModel('LSTM', embedding_dim, class_names, bat_size, num_epochs, vocab_size=voc_size)
# lstm_model.train_model(training_data, validation_data, history_path)
# lstm_model.save_model("lstm_model_4000")

# ----- EVALUATE MODEL ------
# model_path = '/home/myuser/testTweet/LID/saved_model/lstm_model_4000'
#
# test_data = lstm_preprocess.tokenize('/home/myuser/testTweet/LID/test_data.csv', tokenizer)
# lstm_evaluation = EvaluateModel(model_path, validation_data, test_data,)
# lstm_evaluation.evaluate_model()

# norweigan_line_2 = "Jeg synes det er gøy med is"
# swedish_line = "В Москве до смерти избили битами водителя. Дорожный конфликт.  Ад какой-то. Дикая страна"
# norweigan_line = 'vi er like'
# norw_line = lstm_preprocess.tokenize_line(norweigan_line, tokenizer)
# swe_line = lstm_preprocess.tokenize_line(swedish_line, tokenizer)
# print(norweigan_line, ', prediction:', lstm_evaluation.predict_line(norw_line))
# print(swedish_line, ', prediction:', lstm_evaluation.predict_line(swe_line))

# print("Time to predict 8000 tweets for LSTM: ", lstm_evaluation.speed_test(), "seconds")

# Bar chart of accuracy over tweet sample lengths
# lstm_evaluation.plot_bar_chart(x_test=test_data['x_data'], x_test_pad=test_data['x_data_pad'],
#                                y_test=test_data['y_data'], labels=class_names)

# lstm_evaluation.plot_lang_bar(x_test=test_data['x_data'], x_test_pad=test_data['x_data_pad'],
#                               y_test=test_data['y_data'], labels=class_names)
#
#
# lstm_evaluation.plot_language_dis(x_test=test_data['x_data'], x_test_pad=test_data['x_data_pad'],
#                                   y_test=test_data['y_data'], labels=class_names)

# y_pred = lstm_evaluation.predict_data()
# y_actual = test_data["y_data"]
# print(confusion_matrix(y_actual, y_pred))
# print(classification_report(y_actual, y_pred))
# ------ VISUALIZE LSTM MODEL ------

# print("Tweet:", test_data['x_data'][50])
# print("Correct label:", test_data['y_data'][50])
# print("Sigmoid-layer output vector:", y_pred[50])
# print(test_data['x_data'][3], test_data['x_data_pad'][3], test_data['y_data'][3])
# print(test_data['x_data'][49], test_data['x_data_pad'][49], test_data['y_data'][49])
# Plot graph of accuracy and loss of model over number of epochs
# history = load_history(history_path)
# plot_graphs(history)
#
# char_count, max_len, min_len = char_count(lstm_preprocess.data)
# create_histogram(lstm_preprocess.data)
# y_prediction = lstm_evaluation.predict_data()
# create_confusion(test_data['y_data'], y_prediction)
# save_confusion(y_actual, y_pred, class_names)

# ------ fastText model ------
fast_text = FastText('training_data_fasttext_9.txt', 'test_data_fasttext_9.txt')
fast_text_model = fast_text.train_fast_text_model()
y_actual, y_pred = fast_text.get_test_pred(fast_text_model)


for i in range(len(y_actual)):
    text = re.sub('__label__', '', y_actual[i])
    y_actual[i] = text

for i in range(len(y_pred)):
    new = re.sub('__label__', '', y_pred[i])
    y_pred[i] = new


save_confusion(y_actual, y_pred, class_names)
print("Time to predict 4000 tweets for fastText: ", fast_text.speed_test(fast_text_model), "seconds")

print(confusion_matrix(y_actual, y_pred))
print(classification_report(y_actual, y_pred))



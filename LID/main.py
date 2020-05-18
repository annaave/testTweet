from LID.preprocess import Preprocess

class_names = ['English', 'Swedish', 'Spanish', 'Portuguese', 'Russian']
vocab_size = 300
lstm_preprocess = Preprocess('2000_5_data_files_LID.csv', class_names, vocab_size)

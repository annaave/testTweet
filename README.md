#Automatic language identification project

A project that explores how to retrieve tweets through Twitter's API and use them to create and train a LSTM model and Facebook's fastText model. The project consists of 6 Python classes/files; getTweets.py, preprocess.py, train_model.py, evaluate.py, fasttext_2.py and visualize.py. The main file demonstrates how to use a couple of the classes and methods. 

getTweet.py: A Python file that uses the open source library tweepy to access tweets from its tweet identification number. A Twitter developer account is needed to get the credential to take use of Twitter's API.

preprocess.py: A Python class that read the text data from a txt-file with the names of the files that is wanted to be used in the projects. The data is then shuffelsed and can be divided into the training, validation and text dataset, cleaned from addresses and emojis and tokenized.

train_model.py: A Python class that creates a LSTM model and train it.

evaluate.py: A Python class that gives the results and evaluation of the trained machine learning model, create figures to illustarte the result of the model.

fasttext_2.py. A Python class that creates a training and test dataset in the correct form for a fastText model to be trained by.

visualize.py: A Python file that creates figures from the training history for the machine learning model and creating a visual confusion matrix. 


Using the Twitter corpus by:
{Mozetič, Igor; Grčar, Miha and Smailović, Jasmina, 2016, Twitter sentiment for 15 European languages, Slovenian language resource repository CLARIN.SI, http://hdl.handle.net/11356/1054.}

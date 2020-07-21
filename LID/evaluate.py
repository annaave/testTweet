import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


class EvaluateModel:
    """A class that evaluates an already trained and loaded machine learning model. The class has several methods
    to saved plots of the evaluation and gives numbers such as accuracy and loss for the restored model, saved history
    of training and predicting new datasets and evaluating them too."""

    def __init__(self, model, validation_data, test_data):
        self.model = tf.keras.models.load_model(model)
        self.validation_data = validation_data
        self.test_data = test_data
        # Check the model architecture
        self.model.summary()

    def evaluate_model(self):
        # Evaluate the restored model
        loss, acc = self.model.evaluate(self.validation_data['x_data_pad'], self.validation_data['y_data'], verbose=1)
        print("Loss: %.2f" % loss)
        print('Restored model, validation accuracy: {:5.2f}%'.format(100 * acc))
        # Evaluate the restored model
        loss2, acc2 = self.model.evaluate(self.test_data['x_data_pad'], self.test_data['y_data'], verbose=1)
        print("Loss: %.2f" % loss2)
        print('Restored model, test accuracy: {:5.2f}%'.format(100 * acc2))

    # Predict a new dataset and print out the predictions of the 10 first
    def predict_data(self):
        print(self.test_data['x_data'][:10])
        print(self.test_data['y_data'][:10])
        y_pred = self.model.predict_classes(self.test_data['x_data_pad'])
        print(y_pred[:10])
        return y_pred

    # Predict one new line of text
    def predict_line(self, line):
        y_pred = self.model.predict(line)
        return y_pred

    def speed_test(self):
        start = time.time()
        for i in range(4):
            self.model.predict_classes(self.test_data['x_data_pad'], batch_size=len(self.test_data['x_data_pad']))
        time_taken = (time.time() - start)
        return time_taken

    # Saves a bar chart figure showing the accuracy of predicted text towards the length of the text.
    def plot_bar_chart(self, x_test, y_test, x_test_pad, labels):

        # Extracting the character lengths of the input data texts.
        length_test = []
        for i in range(len(x_test)):
            length_test.append(len(x_test[i]))
        print(length_test)
        print(len(length_test))

        count_10 = []
        count_20 = []
        count_40 = []
        count_60 = []
        count_80 = []
        count_100 = []
        count_120 = []

        for i in range(len(length_test)):
            if length_test[i] <= 20:
                count_10.append(i)
            if 20 < length_test[i] <= 40:
                count_20.append(i)
            if 40 < length_test[i] <= 60:
                count_40.append(i)
            if 60 < length_test[i] <= 80:
                count_60.append(i)
            if 80 < length_test[i] <= 100:
                count_80.append(i)
            if 100 < length_test[i] <= 120:
                count_100.append(i)
            if length_test[i] > 120:
                count_120.append(i)

        print(len(count_10) + len(count_20) + len(count_40) + len(count_60) + len(count_80) + len(count_100) + len(
            count_120))

        # Dividing the data in 7 bins with different character length of the input data.
        objects = ('0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140')
        count = [count_10, count_20, count_40, count_60, count_80, count_100, count_120]

        predictions = [[] for _ in range(len(count))]
        new_predictions = [[] for _ in range(len(count))]
        true_positives = []
        accuracy = []

        # Predicting the data and number of correct predictions to the corresponding text length bin.
        for i in range(0, len(count)):
            print(count[i])
            print([labels[label] for label in y_test[count[i]]])
            predictions[i] = self.model.predict(x_test_pad[count[i]])
            true_positives.append(0)
            for j in range(len(count[i])):
                a = np.argmax(predictions[i][j])
                new_predictions[i].append(a)
                if new_predictions[i][j] == y_test[count[i][j]]:
                    true_positives[i] = true_positives[i] + 1

            # Printing each bin of character length of text and the accuracy of that bin.
            print([labels[label] for label in new_predictions[i]])
            print("Number of tweets of character length", objects[i], ":", len(predictions[i]))
            accuracy.append(true_positives[i] / len(predictions[i]))
            print("Accuracy for tweet lengths", objects[i], "characters:", accuracy[i])

        print()

        y_pos = np.arange(len(objects))

        # Plotting each bin towards every bins' accuracy.
        performance = [accuracy[0], accuracy[1], accuracy[2], accuracy[3], accuracy[4], accuracy[5], accuracy[6]]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.xlabel('Character length of tweet')
        plt.title('Accuracy for different character lengths of tweets')
        plt.savefig('/home/myuser/testTweet/LID/figures/bar_chart.png')
        plt.close()

    # Saving three different plots that gives; the accuracy per language, the softmax output vector value of correctly
    # predicted samples and the mean value of the softmax output vector value of correct prediction towards different
    # lengths of texts.
    def plot_lang_bar(self, x_test, y_test, x_test_pad):

        count_eng = []
        count_swe = []
        count_spa = []
        count_por = []
        count_rus = []
        count_ger = []
        count_pol = []
        count_ser = []

        # Dividing the data into lists for each language.
        for i in range(len(y_test)):
            if y_test[i] == 0:
                count_eng.append(i)
            if y_test[i] == 1:
                count_swe.append(i)
            if y_test[i] == 2:
                count_spa.append(i)
            if y_test[i] == 3:
                count_por.append(i)
            if y_test[i] == 4:
                count_rus.append(i)
            if y_test[i] == 5:
                count_ger.append(i)
            if y_test[i] == 6:
                count_pol.append(i)
            if y_test[i] == 7:
                count_ser.append(i)

        count = [count_eng, count_swe, count_spa, count_por, count_rus, count_ger, count_pol, count_ser]

        objects = ('Eng', 'Swe', 'Spa', 'Por', 'Rus', 'Ger', 'Pol', 'Ser')
        predictions = [[] for _ in range(len(y_test))]
        new_predictions = [[] for _ in range(len(y_test))]
        true_positives = []
        mis_classifications = []
        accuracy = []
        value_prediction = []
        prediction_length = []
        tweet_dist = []

        # Predicting and saving the number of correctly classified samples, the incorrectly classified samples, the
        # length of the sample and the softmax output vector value for the samples, for each language.
        for i in range(0, len(count)):
            predictions[i] = self.model.predict(x_test_pad[count[i]])
            true_positives.append(0)
            mis_classifications.append(0)

            for j in range(len(count[i])):
                a = np.argmax(predictions[i][j])
                new_predictions[i].append(a)

                if new_predictions[i][j] == y_test[count[i][j]]:
                    true_positives[i] = true_positives[i] + 1

                    value = max(predictions[i][j])
                    value_prediction.append(value)
                    prediction_length.append(len(x_test[count[i][j]]))
                    tweet_dist.append(x_test[count[i][j]])

                else:
                    mis_classifications[i] = mis_classifications[i] + 1
                    print("tweet:", x_test[count[i][j]], ": Real label:", y_test[count[i][j]], ": Predicted label:"
                          , new_predictions[i][j], predictions[i][j])

            # Printing the number of texts, correct classification and incorrect classification for each language.
            print("Number of tweets of language", objects[i], ":", len(predictions[i]))
            accuracy.append(true_positives[i] / len(predictions[i]))
            print("Accuracy for", objects[i], ":", accuracy[i])
            print("Number of misclassifications:", mis_classifications[i])
            print("Number of correct classifications:", true_positives[i])

            print()
        data_dist = {"Probability of prediction": value_prediction, "Tweet": tweet_dist,
                     "Length of tweet": prediction_length}
        distribution_probabilities = pd.DataFrame(data_dist)
        distribution_probabilities = distribution_probabilities.sort_values(by="Probability of prediction")
        print(distribution_probabilities)
        print()

        y_pos = np.arange(len(objects))
        performance = [accuracy[0], accuracy[1], accuracy[2], accuracy[3], accuracy[4], accuracy[5], accuracy[6],
                       accuracy[7]]

        # Plotting a bar chart of accuracy for each language.
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.xlabel('Language of tweet')
        plt.title('Accuracy for different languages of tweets')
        plt.savefig('/home/myuser/testTweet/LID/figures/bar_chart_languages.png')
        plt.close()

        # Plotting the softmax output vector value of all correctly predicted samples, ordered.
        distribution_probabilities = distribution_probabilities.reset_index(drop=True)
        print(distribution_probabilities)
        plt.plot(distribution_probabilities.index, distribution_probabilities["Probability of prediction"], 'o')
        plt.xlabel('Index of sample (ordered)')
        plt.ylabel('Probability of the predicted languge')
        plt.title('Correctly classified languages.')
        plt.savefig("/home/myuser/testTweet/LID/figures/prob_correct.png")
        plt.close()

        # Plotting the mean value of the softmax output vector value of correct prediction towards different lengths
        # of texts.
        distribution_probabilities = distribution_probabilities.groupby('Length of tweet').mean()
        print(distribution_probabilities.index)
        plt.plot(distribution_probabilities.index, distribution_probabilities["Probability of prediction"], 'o')
        plt.xlabel('Length of tweet')
        plt.ylabel('Probability from the softmax output vector')
        plt.title('Correctly classified languages.')
        plt.savefig("/home/myuser/testTweet/LID/figures/distribution_prob.png")
        plt.close()

    # Saving a plot of the softmax output vector value from all data of ONE language.
    def plot_language_dis(self, x_test, y_test, x_test_pad, labels):
        count_lang = []

        # Here the language is chosen, the number after == corresponds to the chosen language.
        for i in range(len(y_test)):
            if y_test[i] == 0:
                count_lang.append(i)

        predictions = [[] for _ in range(len(count_lang))]
        new_predictions = [[] for _ in range(len(count_lang))]
        true_positives = []
        mis_classifications = []
        value_prediction = []
        prediction_length = []
        tweet_dist = []

        # Vector of probabilities of 8 languages
        predictions = self.model.predict(x_test_pad[count_lang])

        for i in range(len(count_lang)):
            true_positives.append(0)
            mis_classifications.append(0)

            a = np.argmax(predictions[i])
            new_predictions[i].append(a)

            value = predictions[i][new_predictions[i]]
            value_prediction.append(value)
            prediction_length.append(len(x_test[count_lang[i]]))

            if new_predictions[i] == y_test[count_lang[i]]:
                true_positives[i] = true_positives[i] + 1
            else:
                mis_classifications[i] = mis_classifications[i] + 1

        print("Number of test samples for this languge:", len(value_prediction))
        cor_index = list(range(0, len(value_prediction)))
        data = {"proba": value_prediction}
        probabilities_correct = pd.DataFrame(data)
        probabilities_correct = probabilities_correct.sort_values(by="proba")
        probabilities_correct = probabilities_correct.reset_index(drop=True)
        probabilities_correct.insert(loc=0, column='index', value=cor_index)
        print(probabilities_correct)

        # Plot of the softmax output vector value for every correctly classified sample for this one language
        # in dataset.
        # plt.plot(probabilities_correct.index, probabilities_correct["proba"], 'o')
        # plt.xlabel('Index of sample (ordered)')
        # plt.ylabel('Maximum of prediction')
        # plt.title('Serbian tweets.')
        # plt.savefig("/home/myuser/testTweet/LID/figures/all_classified_lang/prob_Ser.png")
        # plt.close()

        # Extracting the softmax output vector value for every sample in dataset for the one language.
        predictions_all = self.model.predict(x_test_pad)
        all_pred_lang = []
        for row in predictions_all:
            all_pred_lang.append(row[0])  # HERE again the number for the language needs to be changed.
        all_pred_lang.sort()
        all_index = list(range(0, len(all_pred_lang)))
        all_pred = {"index": all_index, "proba": all_pred_lang}
        tot = pd.DataFrame(all_pred)
        correct_index = []
        correct_values = []
        for i in range(len(tot)):
            for j in range(len(probabilities_correct)):
                if tot["proba"][i] == probabilities_correct["proba"][j]:
                    correct_index.append(i)
                    correct_values.append(probabilities_correct["proba"][j])
        correct_data = {"index": correct_index, "proba": correct_values}
        correct = pd.DataFrame(correct_data)
        print("Correctly classified samples for this langugae:", correct)
        print(tot)

        new_df = pd.concat([tot, correct]).drop_duplicates(subset=['index'], keep=False)
        print(new_df)
        new_df2 = pd.concat([tot, probabilities_correct])
        new_df2.sort_values(by="proba")
        new_df3 = pd.concat([tot, new_df2]).drop_duplicates(subset=['proba'], keep=False)
        print(new_df3)


        # Plotting the softmax output vector value for every sample for one language, ordered.
        # plt.plot(tot.index, tot["proba"], 'o')
        plt.plot(new_df['index'], new_df["proba"], 'o')
        plt.plot(correct['index'], correct["proba"], 'rx')
        plt.xlabel('Index of sample (ordered)')
        plt.ylabel('Value of prediction')
        plt.title('All outputs for the English probability of all test data samples.')
        plt.savefig("/home/myuser/testTweet/LID/figures/8_languages_pres/all_prob_Eng_index.png")
        plt.close()

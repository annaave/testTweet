import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


class EvaluateModel:
    def __init__(self, model, validation_data, test_data):
        self.model = tf.keras.models.load_model(model)
        self.validation_data = validation_data
        self.test_data = test_data
        # Check its architecture
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

    def predict_data(self):
        print(self.test_data['x_data'][:10])
        print(self.test_data['y_data'][:10])
        y_pred = self.model.predict_classes(self.test_data['x_data_pad'])
        print(y_pred[:10])
        return y_pred

    def predict_line(self, line):
        y_pred = self.model.predict(line)
        return y_pred

    def speed_test(self):
        start = time.time()
        for i in range(4):
            self.model.predict_classes(self.test_data['x_data_pad'], batch_size=len(self.test_data['x_data_pad']))
        time_taken = (time.time()-start)
        return time_taken

    def plot_bar_chart(self, x_test, y_test, x_test_pad, labels):
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
        objects = ('0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140')
        count = [count_10, count_20, count_40, count_60, count_80, count_100, count_120]

        predictions = [[] for _ in range(len(count))]
        new_predictions = [[] for _ in range(len(count))]
        true_positives = []
        accuracy = []
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
            print([labels[label] for label in new_predictions[i]])
            print("Number of tweets of character length", objects[i], ":",  len(predictions[i]))
            accuracy.append(true_positives[i] / len(predictions[i]))
            print("Accuracy for tweet lengths", objects[i], "characters:", accuracy[i])

        # --------------------------------------------------------------
        print()

        y_pos = np.arange(len(objects))

        performance = [accuracy[0], accuracy[1], accuracy[2], accuracy[3], accuracy[4], accuracy[5], accuracy[6]]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.xlabel('Character length of tweet')
        plt.title('Accuracy for different character lengths of tweets')
        plt.savefig('/home/myuser/testTweet/LID/figures/bar_chart.png')
        plt.close()

    def plot_lang_bar(self, x_test, y_test, x_test_pad, labels):

        count_eng = []
        count_swe = []
        count_spa = []
        count_por = []
        count_rus = []
        count_ger = []
        count_pol = []
        count_ser = []
        count_cro = []

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
            if y_test[i] == 8:
                count_cro.append(i)

        count = [count_eng, count_swe, count_spa, count_por, count_rus, count_ger, count_pol, count_ser, count_cro]

        objects = ('Eng', 'Swe', 'Spa', 'Por', 'Rus', 'Ger', 'Pol', 'Ser', 'Cro')
        predictions = [[] for _ in range(len(y_test))]
        new_predictions = [[] for _ in range(len(y_test))]
        true_positives = []
        mis_classifications = []
        accuracy = []
        value_prediction = []
        prediction_length = []
        tweet_dist = []

        for i in range(0, len(count)):
            #print(count[i])
            #print([labels[label] for label in y_test[count[i]]])
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
                    mis_classifications[i] = mis_classifications[i] +1
                    print("tweet:", x_test[count[i][j]], ": Real label:", y_test[count[i][j]], ": Predicted label:"
                          , new_predictions[i][j], predictions[i][j])
            #print([labels[label] for label in new_predictions[i]])
            print("Number of tweets of language", objects[i], ":",  len(predictions[i]))
            accuracy.append(true_positives[i] / len(predictions[i]))
            print("Accuracy for", objects[i], ":", accuracy[i])
            print("Number of misclassifications:", mis_classifications[i])
            print("Number of correct classifications:", true_positives[i])

            print()
        data_dist = {"Probability of prediction": value_prediction, "Tweet": tweet_dist, "Length of tweet": prediction_length}
        distribution_probabilities = pd.DataFrame(data_dist)
        #distribution_probabilities = distribution_probabilities.sort_values(by="Length of tweet")
        distribution_probabilities = distribution_probabilities.sort_values(by="Probability of prediction")
        print(distribution_probabilities)
            # --------------------------------------------------------------
        print()


        y_pos = np.arange(len(objects))
        performance = [accuracy[0], accuracy[1], accuracy[2], accuracy[3], accuracy[4], accuracy[5], accuracy[6], accuracy[7], accuracy[8]]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.xlabel('Language of tweet')
        plt.title('Accuracy for different languages of tweets')
        plt.savefig('/home/myuser/testTweet/LID/figures/bar_chart_languages_9.png')
        plt.close()

        distribution_probabilities = distribution_probabilities.reset_index(drop=True)
        #distribution_probabilities.plot(x="Length of tweet", y="Probability of prediction", logx=True, style='o')
        #distribution_probabilities = distribution_probabilities.groupby('Length of tweet').mean()
        print(distribution_probabilities)
        plt.plot(distribution_probabilities.index, distribution_probabilities["Probability of prediction"], 'o')
        plt.xlabel('Index of sample (ordered)')
        plt.ylabel('Probability of the predicted languge')
        plt.title('Correctly classified languages.')
        plt.savefig("/home/myuser/testTweet/LID/figures/prob_correct_9.png")
        plt.close()

        # distribution_probabilities.plot(x="Length of tweet", y="Probability of prediction", logx=True, style='o')
        distribution_probabilities = distribution_probabilities.groupby('Length of tweet').mean()
        print(distribution_probabilities)
        plt.plot(distribution_probabilities.index, distribution_probabilities["Probability of prediction"], 'o')
        plt.xlabel('Length of tweet')
        plt.ylabel('Probability of the predicted languge')
        plt.title('Correctly classified languages.')
        plt.savefig("/home/myuser/testTweet/LID/figures/distribution_prob_9.png")
        plt.close()

    def plot_language_dis(self, x_test, y_test, x_test_pad, labels):
        count_lang = []

        for i in range(len(y_test)):
            if y_test[i] == 7:
                count_lang.append(i)

        predictions = [[] for _ in range(len(count_lang))]
        new_predictions = [[] for _ in range(len(count_lang))]
        true_positives = []
        mis_classifications = []
        value_prediction = []
        prediction_length = []
        tweet_dist = []

        predictions = self.model.predict(x_test_pad[count_lang]) #vector of probabilities of 8 languages

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

                # value = predictions[i][new_predictions[i]]
                # value_prediction.append(value)
                # prediction_length.append(len(x_test[count_lang[i]]))

            else:
                mis_classifications[i] = mis_classifications[i] + 1
        #print("Tweet:", x_test[count_lang[0]], "prediction:", predictions[0], "max value:", new_predictions[0], value_prediction[0])
        print("Number of test samples for this languge:", len(value_prediction))

        data = {"proba": value_prediction}
        probabilities_correct = pd.DataFrame(data)
        probabilities_correct = probabilities_correct.sort_values(by="proba")
        probabilities_correct = probabilities_correct.reset_index(drop=True)

        # for i in range(len(value_prediction)):
        #     if value_prediction[i] >= [0.9]:
        #         #print("Proba's over 90%:", value_prediction[i], x_test[count_lang[i]])
        #         print()
        #
        #     else:
        #         print()
        #         print("Proba's under 90%:", value_prediction[i], x_test[count_lang[i]])

        plt.plot(probabilities_correct.index, probabilities_correct["proba"], 'o')
        plt.xlabel('Index of sample (ordered)')
        plt.ylabel('Maximum of prediction')
        plt.title('Serbian tweets.')
        plt.savefig("/home/myuser/testTweet/LID/figures/all_classified_lang/prob_Ser_9.png")
        plt.close()

        #------ S-shaped plot --------
        predictions_all = self.model.predict(x_test_pad)
        all_pred_lang = []
        for row in predictions_all:
            all_pred_lang.append(row[7])
        all_pred_lang.sort()
        all_pred = {"pred": all_pred_lang}
        tot = pd.DataFrame(all_pred)
        correct_index = []
        correct_values = []
        for i in range(len(tot)):
            for j in range(len(probabilities_correct)):
                if tot["pred"][i] == probabilities_correct["proba"][j]:
                    correct_index.append(i)
                    correct_values.append(probabilities_correct["proba"][j])
        correct_data = {"index": correct_index, "proba": correct_values}
        correct = pd.DataFrame(correct_data)
        print("Correctly classified samples for this langugae:", correct)
        print(tot)
        plt.plot(tot.index, tot["pred"], 'o')
        #plt.plot(correct["index"], correct["proba"], 'om')
        plt.xlabel('Index of sample (ordered)')
        plt.ylabel('Value of prediction')
        plt.title('All outputs for the Serbian probability of all test data samples.')
        plt.savefig("/home/myuser/testTweet/LID/figures/9_languages/all_prob_Ser_index_9.png")
        plt.close()

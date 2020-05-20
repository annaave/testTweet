import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import time


class EvaluateModel:
    def __init__(self, model, validation_data, test_data):
        self.model = tf.keras.models.load_model(model)
        self.validation_data = validation_data
        self.test_data = test_data
        # Check its architecture
        self.model.summary()

    def evaluate_model(self):
        # Evaluate the restored model
        loss, acc = self.model.evaluate(self.validation_data['x_data_pad'], self.validation_data['y_data'], verbose=2)
        print("Loss: %.2f" % loss)
        print('Restored model, validation accuracy: {:5.2f}%'.format(100 * acc))
        # Evaluate the restored model
        loss2, acc2 = self.model.evaluate(self.test_data['x_data_pad'], self.test_data['y_data'], verbose=2)
        print("Loss: %.2f" % loss2)
        print('Restored model, test accuracy: {:5.2f}%'.format(100 * acc2))

    def predict_data(self):
        y_pred = self.model.predict_classes(self.test_data['x_data_pad'])
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
        count = []
        count.append(count_10)
        count.append(count_20)
        count.append(count_40)
        count.append(count_60)
        count.append(count_80)
        count.append(count_100)
        count.append(count_120)

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
        plt.savefig('/home/myuser/testTweet/LID/figures/bar_chart_15epochs.png')
        plt.close()



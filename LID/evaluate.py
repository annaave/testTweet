import tensorflow as tf


class EvaluateModel:
    def __init__(self, model, validation_data, test_data):
        self.model = model
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

    def predict_data(self, x_test_padded):
        y_pred = self.model.predict_classes(x_test_padded)
        return y_pred

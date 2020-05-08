import tensorflow as tf
import LID.train_model as tm


class EvaluateModel:
    def __init__(self, model):
        self.model = model

    def evaluate_model(self, x_test_pad, y_test, x_validation_pad, y_validation):
        new_model = tf.keras.models.load_model(self.model)

        # Check its architecture
        new_model.summary()
        # Evaluate the restored model
        loss, acc = new_model.evaluate(x_validation_pad, y_validation, verbose=2)
        print("Loss: %.2f" % loss)
        print('Restored model, validation accuracy: {:5.2f}%'.format(100 * acc))
        # Evaluate the restored model
        loss2, acc2 = new_model.evaluate(x_test_pad, y_test, verbose=2)
        print("Loss: %.2f" % loss2)
        print('Restored model, test accuracy: {:5.2f}%'.format(100 * acc2))

        print(new_model.predict(x_test_pad).shape)

def main():
    vocab_size = 300
    embedding_dim = 128
    num_epochs = 15
    bat_size = 32
    max_length = 150
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'

    class_names = ["English", "Swedish", "Spanish", "Portuguese", "Russian"]
    lstm_model = tm.TrainModel()
    lstm_model.create_lstm_model(vocab_size, embedding_dim, class_names)
    x_train, x_train_pad, y_train, x_validation, x_validation_pad, y_validation, x_test, x_test_pad, y_test = \
        lstm_model.tokenize(vocab_size, oov_tok, trunc_type, padding_type, max_length)

    new_evaluation = EvaluateModel('/home/myuser/testTweet/LID/saved_model/lstm_model')
    new_evaluation.evaluate_model(x_test_pad, y_test, x_validation_pad, y_validation)


if __name__ == "__main__":
        main()

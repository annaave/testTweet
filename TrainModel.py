
history = model.fit(x_train_pad, y_train, batch_size=64, epochs=num_epochs,
                    validation_data=(x_validation_pad, y_validation), verbose=1, callbacks=[es])

loss, acc = model.evaluate(x_validation_pad, y_validation, verbose=1)
print("Loss: %.2f" % loss)
print("Validation Accuracy: %.2f" % acc)

loss2, acc2 = model.evaluate(x_test_pad, y_test, verbose=1)
print("Loss: %.2f" % loss2)
print("Test Accuracy: %.2f" % acc2)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.axhline(y=0.9, color='r', linestyle='--')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('5_lang_2000_April_28_128.png')
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('5_lang_loss_2000_April_28_128.png')
plt.close()

# y_pred = model.predict_classes(x_test_pad)
y_pred2 = model.predict_classes(x_validation_pad)
# print(tf.math.confusion_matrix(labels=y_test, predictions=y_pred))
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
print(classification_report(y_validation, y_pred2))
print(confusion_matrix(y_validation, y_pred2))


print('\n# Generate predictions for 6 samples from the hold-out dataset (testing set)')
predictions = model.predict(x_test_pad)
print('prediction 1:', x_test[0], predictions[0], "Correct label:", y_test[0])
print('prediction 2:', x_test[1], predictions[1], "Correct label:", y_test[1])
print('prediction 3:', x_test[2], predictions[2], "Correct label:", y_test[2])
print('prediction 4:', x_test[3], predictions[3], "Correct label:", y_test[3])
print('prediction 5:', x_test[4], predictions[4], "Correct label:", y_test[4])
print('prediction 6:', x_test[5], predictions[5], "Correct label:", y_test[5])

# Plot training & validation loss values
length_test = []
for i in range(len(x_test)):
    length_test.append(len(x_test[i]))


new_line = "perfekta tweets - cool"
new_sequences = tokenizer.texts_to_sequences([new_line])
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
new_prediction = model.predict(new_padded)
print('prediction of:', new_line, new_prediction[0], "Correct label: Svenska eller Engelska?")

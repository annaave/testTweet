import fasttext

# https://fasttext.cc/docs/en/language-identification.html

pretrained_model_path = '/home/myuser/testTweet/lid.176.ftz'
model = fasttext.load_model(pretrained_model_path)

sentences = ['vad händer här då']

predictions = model.predict(sentences)
print(predictions)

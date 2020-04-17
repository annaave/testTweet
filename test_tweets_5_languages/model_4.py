#try 2-grams and make all functions general!
#also do it with an embedding layer, even though character n-grams
#ithub.com/CyberZHG/keras-word-char-embd


def encode_ngram_text(text, n):
    text = [text[i:i + n] for i in range(len(text) - n + 1)] #splitting up the text as n-grams
    print(text)
    new_text = []
    for gram in text:
        new_gram = [ord(gram[i]) for i in range(len(gram))]
        new_text.append(new_gram)
    text = new_text
    #text = [ord(text[i]) for i in range(len(text))] #converting every character to its unicode
    return text
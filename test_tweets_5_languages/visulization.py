import pandas as pd
import re
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np


def read_all(class_names):
    max_rows = 10000
    d = dict(zip(class_names, range(0, 5)))
    print(d)

    df1 = (pd.read_csv("2000_Eng_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df1 = df1.drop(index=0)
    df1 = df1.reset_index(drop=True)

    df2 = (pd.read_csv("2000_Swe_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df2 = df2.drop(index=0)
    df2 = df2.reset_index(drop=True)

    df3 = (pd.read_csv("2000_Rus_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df3 = df3.drop(index=0)
    df3 = df3.reset_index(drop=True)

    df4 = (pd.read_csv("2000_Por_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df4 = df4.drop(index=0)
    df4 = df4.reset_index(drop=True)

    df5 = (pd.read_csv("2000_Spa_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df5 = df5.drop(index=0)
    df5 = df5.reset_index(drop=True)

    # df6 = (pd.read_csv("2000_Ger_tweets_label.csv", usecols=['tweets', 'language']).dropna(
    #     subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    # df6 = df6.drop(index=0)
    # df6 = df6.reset_index(drop=True)

    all_data = pd.concat([df1, df2, df3, df4, df5], ignore_index=True, sort=False)

    # Clean tweet data
    for i in range(len(all_data)):
        all_data['tweets'][i] = clean_up(all_data['tweets'][i])

    all_data = all_data.sample(frac=1).reset_index(drop=True)
    all_data['language'] = all_data['language'].map(d, na_action='ignore')

    return all_data

def clean_up(line):
    # remove url
    line = re.sub(r'http\S+', '', line)

    # lowercase all letters
    #words = line.split()
    #words = [word.lower() for word in words]
    #line = ' '.join(words)

    # remove emojis
    #line = remove_emojies(line)

    # remove excessive signs
    # remove_characters = re.compile('[/(){}\[\]\|.,;:!?"<>^*&%$]')
    # line = remove_characters.sub('', line)
    return line


def remove_emojies(text):
    # Ref: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1#gistcomment-3208085
    # Ref: https://en.wikipedia.org/wiki/Unicode_block
    emojies = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "])")
    text = re.sub(emojies, r'', text)
    return text


def char_count(data):
    max_len = 0
    min_len = len(data['tweets'][0])
    count_char = 0
    lengths = []
    for row in data['tweets']:
        if len(row) > max_len:
            max_len = len(row)
        if min_len > len(row):
            min_len = len(row)
        for element in row:
            count_char += 1
        lengths.append(len(row))
    data['tweet length'] = lengths
    return count_char, max_len, min_len


def main():
    class_names = ["English", "Swedish", "Spanish", "Portuguese", "Russian"]
    pd.options.display.max_colwidth = 2000
    all_data = read_all(class_names)
    chars, max_len, min_len = char_count(all_data)
    rows = len(all_data)

    print("Number of rows:", rows, "Number of characters:", chars)
    print("Maximum length of tweet:", max_len)
    print("minimum lenth of tweet:", min_len)
    print(all_data)
    all_data.to_csv('all_data_tweet_length.csv')

    s = all_data['tweet length']
    ax = s.hist(bins=20, grid=False)
    ax.set_xlabel('Number of characters')
    ax.set_ylabel('Number of tweets')
    ax.set_title('Distribution of tweet lengths')
    fig = ax.get_figure()
    fig.savefig('histogram.png')
    plt.close()

    sn.distplot(all_data['tweet length'])
    plt.savefig('pairplot.png')
    plt.close()

    plt.scatter(all_data['tweet length'], all_data['language'])
    plt.savefig('scatter.png')
if __name__ == "__main__":
    main()

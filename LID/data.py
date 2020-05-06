import pandas as pd


def read_all(class_names):
    max_rows = 10000
    d = dict(zip(class_names, range(0, 6)))
    print(d)
    pd.options.display.max_colwidth = 1000

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

    df6 = (pd.read_csv("2000_Ger_tweets_label.csv", usecols=['tweets', 'language']).dropna(
        subset=['tweets', 'language']).assign(tweets=lambda x: x.tweets.str.strip()).head(max_rows))
    df6 = df6.drop(index=0)
    df6 = df6.reset_index(drop=True)

    all_data = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True, sort=False)

    all_data = all_data.sample(frac=1).reset_index(drop=True)
    all_data['language'] = all_data['language'].map(d, na_action='ignore')

    print(all_data)
    print(len(all_data))
    return all_data


def main():
    print('hej!')


if __name__ == "__main__":
    main()

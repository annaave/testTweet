import tweepy
import pandas as pd
import time

consumer_key = "xxxxx"
consumer_secret = "xxxxx"
access_token = "xxxxx"
access_token_secret = "xxxxx"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
# test authentication
#try:
#    api.verify_credentials()
#    print("Authentication OK")
#except:
#    print("Error during authentication")
#

def print_tweets(file, count, newFile):
    id_tweet = []
    list_tweets = []
    data = pd.read_csv(file)

    # Removing duplicates of Twitter IDs in the CSV file.
    print("Rows of data before duplicated rows deleted:", data.size)
    data.drop_duplicates(subset="TweetID", keep=False, inplace=True)
    print("Rows of data before duplicated rows deleted:", data.size)

    #for i in range(count):
     #   id_tweet.append(data["TweetID"].iloc[i])
    #print(id_tweet)
    #tweet = api.get_status(swe_data["TweetID"].iloc[20286])
    for i in range(data.size):
        try:
            #print(i+1)
            id_tweet.append(data["TweetID"].iloc[i])
            tweet = api.get_status(id_tweet[i], tweet_mode='extended')
            list_tweets.append(tweet.full_text)
        except Exception as e:
            if e == "[{u'message': u'Rate limit exceeded', u'code': 88}]":
                print("Will g to sleep for 5 minutes now!")
                time.sleep(60*5) #Sleep for 5 minutes
            else:
                print(e)
                pass
        if len(list_tweets) == count:
            break
    print("Size of list with data including unusable TweetID:s:", len(id_tweet))
    print("Number of collected tweet texts:", len(list_tweets))

    for i in range(len(list_tweets)):
        print("Tweet:", list_tweets[i])

    df = pd.DataFrame(list_tweets)
    df.to_csv(newFile, index=False)
#def clean_tweets(tweets):
 #   tweet.text = tweet.text.split(' ',1)[1]
  #  print(tweet.text)

#tweet = api.statuses_lookup(id_swe)
#print(tweet.text)
#print(len(tweet.text))


#name =
#random_tweet =
#print(name, ':', api.get_status(random_tweet).text)


#Make call on home (my own) timeline, print each tweets text
#public_tweets = api.home_timeline()
#for tweets in public_tweets:
#    print(tweets.text)



#------ NOT SURE HOW THE REPLIES METHODS WORK ------
def replies_tweet_1(name, tweet_id, number):
    replies = []
    for tweet in tweepy.Cursor(api.search, q='to:' + name, result_type='recent', timeout=999999).items(number):
        if hasattr(tweet, 'in_reply_to_status_id_str'):
            if (tweet.in_reply_to_status_id_str == tweet_id):
                replies.append(tweet.text)
    for i in range(len(replies)):
        print("Reply:", replies[i])

def replies_tweet_2(name, tweet_id, number):
    replies = []
    # non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
    limit = 50

    for full_tweets in tweepy.Cursor(api.user_timeline, screen_name=name, timeout=999999).items(limit):
        for tweet in tweepy.Cursor(api.search, q='to:' + name, since_id=tweet_id, result_type='recent',
                                   timeout=999999).items(number):
            if hasattr(tweet, 'in_reply_to_status_id_str'):
                if (tweet.in_reply_to_status_id_str == full_tweets.id_str):
                    replies.append(tweet.text)
        # print("Tweet :",full_tweets.text.translate(non_bmp_map))
        for elements in replies:
            print("Replies :", elements)

def main():
    print("Portuguese tweets:")
    print_tweets("Portuguese_Twitter_sentiment.csv", 50, "new_Port_tweets.csv")

    #print("English tweets:")
    #print_tweets("English_Twitter_sentiment.csv", 5)


if __name__ == "__main__":
    main()




import tweepy
import csv
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import argparse
import pandas as pd
import datetime as time
import os
from gemlib.abstarct.basefunctionality import BaseDataLoader
from gemlib.classification.topicmodelling.tweets.preprocess import TwitterPreprocessing

consumer_key = 'N0GtV5tWVp04RSBMWfa15egKa'
consumer_secret = 'KVCEp1rm2W5HYD3aerJodgIy8iEH8L1Q5OZvdgWsA41rRzJr9F'
access_token = '789068201883602944-yUIhxzGymCHrVU1K5Zxc0qkqxtCYqnx'
access_token_secret = 'FbMareIEtDHQPxBIv4r0Q3qRwNvk9ULXQd6Qub9fE16cF'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)
location_bbox=[110.917969,-39.300299,156.796875,-10.919618]
columns = ['tweet.id_str',
            'tweet.created_at',
            'tweet_text',
            'tweet.geo',
            'tweet.coordinates',
            'tweet.place',
            'tweet.retweet_count',
            'tweet.favorite_count',
            'tweet.retweeted',
            'tweet.source',
            'tweet.user.id_str',
            'tweet.user.location',
            'tweet.user.follower_count',
            'tweet.user.friends_count',
            'tweet.user.geo_enabled']


# streaming
class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    counter = 0
    list_tweets = []
    streamfilename = ''
    tweet_count_threshold = 1000

    def on_status(self, tweet):
        if tweet.retweeted:
            return

        # if ('event' not in status.text.lower()) and ('concert' not in status.text.lower()) :
        #             return


        if hasattr(tweet, 'retweeted_status'):
            try:
                text = tweet.retweeted_status.extended_tweet["full_text"]
            except:
                text = tweet.retweeted_status.text
        else:
            try:
                text = tweet.extended_tweet["full_text"]
            except AttributeError:
                text = tweet.text

        tweet_dict = tweet._json
        fields = [tweet.id_str,
                  tweet.created_at,
                  str(text.encode('utf8')),
                  tweet.geo,
                  tweet.coordinates,
                  str(tweet_dict['place']),
                  tweet.retweet_count,
                  tweet.favorite_count,
                  tweet.retweeted,
                  tweet.source,
                  tweet.user.id_str,
                  str(tweet_dict['user']['location']),
                  tweet.user.followers_count,
                  tweet.user.friends_count,
                  tweet.user.geo_enabled]

        self.counter += 1
        self.list_tweets.append(fields)
        if self.counter % 10 == 0:
            dt = time.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            print(f'Total tweets until {dt}:{self.counter}', flush=True)
            if self.counter % self.tweet_count_threshold == 0:
                df = pd.DataFrame(self.list_tweets, columns=columns)
                df.fillna('null', inplace=True)
                self.list_tweets = []
                # preprocessing
                preproc = TwitterPreprocessing(bins=0.2)
                df = preproc.apply(df)
                df.to_csv(self.streamfilename, mode='a', header=False, index=False)

        return True

    def on_error(self, status_code):
        if status_code == 420:
            return False

class TweetStreamer(BaseDataLoader):

    def start_stream(self, auth, listener):
        while True:
            try:
                stream = Stream(auth, listener)
                stream.filter(locations=location_bbox)
            except Exception as exception:
                Logging.log_exception(exception, False)
                print('restarting the stream...')
                continue

    def create_output_header(self):
        # create a new streaming file
        with open(self.path, 'w') as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(columns)

    def stream_runner(self):
        # streaming runner
        listener = StdOutListener()
        listener.streamfilename = self.path
        listener.tweet_count_threshold = self.tweet_counter
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        if not os.path.isfile(self.path):
            self.create_output_header(self.path)
        self.start_stream(auth, listener)

    def load(self):
        self.stream_runner()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tweet project streaming')
    parser.add_argument('-f', metavar='filePath', type=str)
    parser.add_argument('-c', metavar='tweetThr', type=int, default=1000)
    args = parser.parse_args()
    streaming = TweetStreamer(path=args.f, tweet_counter=args.c)
    streaming.load()
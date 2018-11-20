import os
import sys
import pickle
import sqlite3
from datetime import datetime
from tweetscrape.profile_tweets import TweetScrapperProfile


class CollectUser:
    def __init__(self, sqlite_file, sqlite_table, user_file, npage = 2, e_dir = '../../data/'):
        self.sqlite_file = sqlite_file
        self.sqlite_table = sqlite_table
        self.user_file = user_file
        self.npage = npage
        self.temp_msg = {}
        self.e_dir = e_dir
        self.e_mentions = []
        self.e_hashtags = []

    def cast_unix(self, strdate):
        return(str(datetime.fromtimestamp(int(strdate[0:10])).strftime('%Y-%m-%d %H:%M:%S')))

    def fetch_profiles(self, filename):
        f           = open(filename, 'r')
        profiles    = f.read().splitlines()
        f.close()
        return(list(set(profiles)))

    def build_dict(self, TweetInfo):
        result = {}
        result['Id']        = TweetInfo.get_tweet_id()
        result['Time']      = '"{}"'.format(self.cast_unix(TweetInfo.get_tweet_time_ms()))
        result['Author']    = '"{}"'.format(TweetInfo.get_tweet_author())
        result['Text']      = '"{}"'.format(TweetInfo.get_tweet_text())
        result['Hashtags']  = '"{}"'.format(TweetInfo.get_tweet_hashtags())
        result['Mentions']  = '"{}"'.format(TweetInfo.get_tweet_mentions())
        result['Replies']   = TweetInfo.get_tweet_replies_count()
        result['Favourites']= TweetInfo.get_tweet_favorite_count()
        result['Retweets']  = TweetInfo.get_tweet_retweet_count()
        self.temp_msg['Author'] = result['Author']
        self.temp_msg['Id'] = result['Id']
        self.e_mentions.extend(result['Mentions'])
        self.e_hashtags.extend(result['Hashtags'])
        return(result)

    def build_query(self, table, ddict):
        cols = ''
        vals = ''
        for key, val in ddict.items():
            if (key != list(ddict.keys())[-1]):
                cols += ' {},'.format(key)
                vals += ' {},'.format(val)
            else:
                cols += ' {}'.format(key)
                vals += ' {}'.format(val)
        query = 'INSERT INTO {} ({}) VALUES ({})'.format(table, cols, vals)
        return(query)

    def picklelist(self, item, path):
        with open(path, 'wb') as fp:
            pickle.dump(item, fp)

    def fetch_tweets(self, profile, npage):
        TS = TweetScrapperProfile(profile, npage)
        tweets = TS.get_profile_tweets()
        return(tweets)

    def collect(self):
        profiles = self.fetch_profiles(self.user_file)
        cnxn = sqlite3.connect(self.sqlite_file)
        c = cnxn.cursor()
        for profile in profiles:
            tweets = self.fetch_tweets(profile, self.npage)
            for tweet in tweets:
                q = self.build_query(self.sqlite_table, self.build_dict(tweet))
                try:
                    c.execute(q)
                except sqlite3.OperationalError as e:
                    print(e, q)
                print("Tweet inserted: {}".format(self.temp_msg))
            cnxn.commit()
        cnxn.close()
        self.picklelist(self.e_hashtags, self.e_dir + 'hashtags.pickle')
        self.picklelist(self.e_mentions, self.e_dir + 'mentions.pickle')

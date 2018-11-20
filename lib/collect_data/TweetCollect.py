import os
import sys
import pickle
import sqlite3
from datetime import datetime
from tweetscrape.profile_tweets import TweetScrapperProfile


class TweetCollectUser(sqlite_file, sqlite_table, user_file, e_dir = '../../data/'):
    def __init__(self):
        self.sqlite_file = sqlite_file
        self.sqlite_table = sqlite_table
        self.user_file = user_file
        self.temp_msg = {}
        self.e_dir = e_dir
        self.e_mentions = []
        self.e_hashtags = []

    def cast_unix(strdate):
        return(str(datetime.fromtimestamp(int(strdate[0:10])).strftime('%Y-%m-%d %H:%M:%S')))

    def fetch_profiles(filename):
        f           = open(filename, 'r')
        profiles    = f.read().splitlines()
        f.close()
        return(list(set(profiles)))

    def build_dict(TweetInfo):
        result = {}
        result['Id']        = TweetInfo.get_tweet_id()
        result['Time']      = '"{}"'.format(cast_unix(TweetInfo.get_tweet_time_ms()))
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

    def build_query(table, ddict):
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

    def picklelist(item, path):
        with open(path, 'wb') as fp:
            pickle.dump(item, fp)

    def collect(recursive, npage):
        profiles = fetch_profiles(self.user_file)
        cnxn = sqlite3.connect(self.sqlite_file)
        c = cnxn.cursor()
        for profile in profiles:
            tweets = fetch_tweets(profile, npage)
            for tweet in tweets:
                q = build_query(table_name, build_dict(tweet))
                c.execute(q)
                print("Tweet inserted: {}".format{self.temp_msg})
            cnxn.commit()
        cnxn.close()
        picklelist(self.e_hashtags, self.e_dir + 'hashtags.pickle')
        picklelist(self.e_mentions, self.e_dir + 'mentions.pickle')

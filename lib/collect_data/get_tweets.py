import sys
import os
import sqlite3
from datetime import datetime

from tweetscrape.profile_tweets import TweetScrapperProfile

sqlite_file = '../../data/database/deeplearning.sqlite'
table_name = 'tweets'

def cast_unix(strdate):
    return(str(datetime.fromtimestamp(int(strdate[0:10])).strftime('%Y-%m-%d %H:%M:%S')))

""" Get targets """
def fetch_profiles(filename):    
    f           = open(filename, 'r')
    profiles    = f.read().splitlines()
    f.close()
    return(profiles)

""" Get a tweet """
def fetch_tweets(profile, npages = 1):
    TS = TweetScrapperProfile(profile, npages)
    tweets = TS.get_profile_tweets()
    return(tweets)

def db_insert(cursor, query):
    cursor.execute(query)
    return(0)

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

if __name__ == '__main__':    
    profiles = fetch_profiles('profiles.txt')
    cnxn = sqlite3.connect(sqlite_file)
    c = cnxn.cursor()
    for profile in profiles:
        tweets = fetch_tweets(profile, 1)
        for tweet in tweets:
            q = build_query(table_name, build_dict(tweet))
            print(q)
            c.execute(q)
        print("Profile inserted")
        cnxn.commit()
    cnxn.close()








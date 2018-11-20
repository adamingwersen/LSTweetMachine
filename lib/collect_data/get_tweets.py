import TweetCollect as tc

if __name__ == '__main__':
    sqlite_file     = '../../data/database/deeplearning.sqlite'
    sqlite_table    = 'tweets'
    user_file       = '../../data/profiles.txt'
    clct = tc.CollectUser(sqlite_file, sqlite_table, user_file, npage = 2)
    clct.collect()

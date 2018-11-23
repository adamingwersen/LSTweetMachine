import TweetCollect as tc

if __name__ == '__main__':
    sqlite_file     = '../../data/database/deeplearning.sqlite'
    sqlite_table    = 'donald'
    user_file       = '../../data/donald.txt'
    clct = tc.CollectUser(sqlite_file, sqlite_table, user_file, npage = 25)
    clct.collect()

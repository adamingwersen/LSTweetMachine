import pandas as pd
import numpy as np
import sqlite3
import re

class CleanData:
    def __init__(self, sqlite_file, sqlite_table):
        self.sqlite_file = sqlite_file
        self.sqlite_table = sqlite_table
        self.cnxn = None
        self.data = None
        self.query = ""

    def _connect(self):
        self.cnxn = sqlite3.connect(self.sqlite_file)

    def set_table(self, _query):
        self._connect()
        self.data = pd.read_sql_query(_query, self.cnxn)

    def get_clean_table(self, tcol = 'Text', newcol = 'CleanText'):
        self.data[newcol] = self.data[tcol].apply(lambda t: self._strip_links(t))
        self.data[newcol] = self.data[newcol].apply(lambda t: self._strip_ats(t))
        self.data[newcol] = self.data[newcol].apply(lambda t: self._strip_metachar(t))
        self.data[newcol] = self.data[newcol].apply(lambda t: self._strip_whitespace(t))
        self.data[newcol] = self.data[newcol].apply(lambda t: t.lower())
        self.data[newcol] = self.data[newcol].apply(lambda t: self._detect_empty(t))
        self.data = self.data.replace(r'(^\s+$)', np.nan, regex=True)
        self.data = self.data.dropna(subset=[newcol])
        return(self.data)

    def _strip_links(self, txt):
        txt = re.sub(r'(\w+|\.+|\/+)\.twitter.com(\/).*\s', '', txt, flags = re.I)
        txt = re.sub(r'(?:\w+|\@\w+|\#\w+|\s).twitter\.com\/\w*', '', txt, flags = re.I)
        txt = re.sub(r'(?:\w+|\@\w+|\#\w+|\s).twitter.com\w*', '', txt, flags = re.I)
        return(re.sub(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', txt))

    def _strip_whitespace(self, txt):
        txt = txt.strip(' ')
        txt = re.sub('( \- | \-)', '', txt)
        return(re.sub(r' +', ' ', txt))

    def _strip_metachar(self, txt):
        return(re.sub(r"[^a-zA-Z0-9\@\# ]+", '', txt))

    def _strip_ats(self, txt):
        return(re.sub(r'(\@|\#)\w*', '', txt))

    def _detect_empty(self, txt):
        if txt == '':
            return(np.nan)
        else:
            return(txt)

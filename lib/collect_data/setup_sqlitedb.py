import sqlite3

""" Setup Parameters """
sqlite_file     = '../../data/database/deeplearning.sqlite'
table_name      = 'tweets' 
column_names    = ['Id', 'Time', 'Author', 'Text', 'Hashtags', 'Mentions', 'Replies', 'Favourites', 'Retweets']
column_types    = ['INTEGER', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'INTEGER', 'INTEGER', 'INTEGER']

""" Establish connection to db """
cnxn    = sqlite3.connect(sqlite_file)
c       = cnxn.cursor() 

""" Setup our createtable query """
create_query = 'CREATE TABLE {} ('.format(table_name)
for nm, tp in zip(column_names, column_types):
    if nm != column_names[-1]:
        create_query += '{} {}, '.format(nm, tp)
    else:
        create_query += '{} {}'.format(nm, tp)
create_query += ')'

print('Query to be executed: {}'.format(create_query))

""" Execute query """
c.execute(create_query)
cnxn.commit()
cnxn.close()


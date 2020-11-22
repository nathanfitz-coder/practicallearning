import pandas as pd

#dataframe headers
user_header = ["UserID","Gender","Age","Occupation","Zip-code"]
ratings_header = ['UserID','MovieID','Rating','Timestamp']
movies_header = ['MovieID','Title','Genres']

#import data
users = pd.read_csv('MovieData/users.dat',sep='::',names=user_header)
ratings = pd.read_csv('MovieData/ratings.dat',sep='::',names=ratings_header)
movies = pd.read_csv('MovieData/movies.dat',sep='::',names=movies_header)

#split movie genres
movies_genres = pd.concat([pd.Series(row['MovieID'], row['Genres'].split('|')) for _, row in movies.iterrows()]).reset_index().rename(columns={'index': 'genre', 0: 'MovieID'})

tmp = ratings.head()

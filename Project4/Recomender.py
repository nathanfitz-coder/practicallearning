import pandas as pd
import numpy as np

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
movie_grouped = ratings.groupby(by=['MovieID'])['Rating'].agg([np.sum, np.mean, np.std,np.ma.count])

ninty_pct = np.percentile(movie_grouped['count'],90)
meanreview = np.mean(ratings['Rating'])

movie_grouped['wr'] = (movie_grouped['mean']*movie_grouped['count']/(movie_grouped['count']+ninty_pct)) + (meanreview*ninty_pct/(movie_grouped['count']+ninty_pct))

movies = movies.set_index('MovieID')

tmp = movie_grouped.sort_values(by=['wr'], ascending=False).head(10).join(movies)























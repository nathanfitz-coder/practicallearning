import pandas as pd
import numpy as np
import datetime
import sys
import time
import json
from collections import defaultdict
from surprise import SVD, SlopeOne,CoClustering
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from sqlalchemy import create_engine


def movie_weights():
    e = create_engine("sqlite:///../moviereviews.db")
    
    movies = pd.DataFrame(e.execute("SELECT MovieID, Title, Genre FROM movies").fetchall()).rename(columns={0: 'MovieID', 1: 'Title', 2:'Genres'})
    ratings = pd.DataFrame(e.execute("SELECT * FROM ratings").fetchall()).rename(columns={0: 'UserID', 1: 'MovieID', 2:'Rating', 3:'Timestamp'})
    
    movies['Year'] = movies['Title'].apply(lambda x: x[x.rfind("(")+1:-1]).apply(int) 
    
    
    
    firsttime = min(ratings['Timestamp'])
    lasttime = max(ratings['Timestamp'])
    
    def time_to_coef(timestamp):
        return (timestamp-firsttime)/(lasttime-firsttime)
    
    ratings['time_coef'] = ratings['Timestamp'].apply(time_to_coef)
    
    firstyear = min(movies['Year'])
    lastyear = max(movies['Year'])
    
    def year_to_coef(currentyear):
        return (currentyear-firstyear)/(lastyear-firstyear)
    
    movies['year_coef'] = movies['Year'].apply(year_to_coef)
    
    
    
    movies_genres = pd.concat([pd.Series(row['MovieID'], row['Genres'].split('|')) for _, row in movies.iterrows()]).reset_index().rename(columns={'index': 'Genre', 0: 'MovieID'})
    #movies_genres.to_sql('movie_genres', if_exists='replace', con=e)
    
    
    avg_time_coef = ratings.groupby(by=['MovieID'])['time_coef'].agg([np.mean]).rename(columns={'mean': 'avg_time_coef'})
    
    
    movie_grouped = ratings.groupby(by=['MovieID'])['Rating'].agg([np.sum, np.mean, np.std,np.ma.count]).merge(movies, on='MovieID')
    
    ninty_pct = np.percentile(movie_grouped['count'],90)
    fifty_pct = np.percentile(movie_grouped['count'],50)
    
    meanreview = np.mean(ratings['Rating'])
    
    #calculating weighted review
    movie_grouped['weighted_review'] = (movie_grouped['mean']*movie_grouped['count']/(movie_grouped['count']+ninty_pct)) + (meanreview*ninty_pct/(movie_grouped['count']+ninty_pct))
    
    #calculating trendy reviews
    movie_grouped['trendy_review'] = movie_grouped['year_coef']*((movie_grouped['mean']*movie_grouped['count']/(movie_grouped['count']+fifty_pct)) + (meanreview*fifty_pct/(movie_grouped['count']+fifty_pct)))
    
    movie_grouped = movie_grouped.rename(columns={'weighted_review': 'WeightedRating', 'trendy_review': 'TrendRating', 'Genres':'Genre'})
    
    movie_grouped[['MovieID','Title','Genre','WeightedRating','TrendRating']].to_sql('movies', if_exists='replace', con=e,index=False)
    


def best_of_genre_1(genre_rating, g='Animation',n=10):
    genre_filtered = genre_rating[genre_rating['genre']==g]
    return genre_filtered.sort_values(by=['wr'], ascending=False).head(n)['MovieID'].tolist()


def best_of_genre_2(genre_rating, g='Animation',n=10):
    genre_filtered = genre_rating[genre_rating['genre']==g]
    return genre_filtered.sort_values(by=['tr'], ascending=False).head(n)['MovieID'].tolist()




def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n




def cf_recommend(ratings, algo, user_id):
    # First train an SVD algorithm on the movielens dataset.
    #data = Dataset.load_builtin('ml-100k')
    reader = Reader()
    data = Dataset.load_from_df(ratings, reader)
    trainset = data.build_full_trainset()
    
    algo.fit(trainset)
    
    user_ratings=ratings[ratings['UserID']==user_id]
    user_ratings=user_ratings[['UserID', 'MovieID', 'Rating']]
    
    
    #all_items = list(trainset.all_items())
    #fake_user = [9999] * len(all_items)
    #fake_rating = [4.5] * len(all_items)
    
    
    ratings_dict = {'UserID': [9999] * len(trainset.all_items()),
                    'MovieID': list(trainset.all_items()),
                    'Rating': [4.5] * len(trainset.all_items())}
    
    user_ratings=pd.concat([user_ratings, pd.DataFrame(ratings_dict)])
    
    user_ratings = Dataset.load_from_df(user_ratings, reader).build_full_trainset()
    
    
    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = user_ratings.build_anti_testset()
    predictions = algo.test(testset)
    
    top_n = get_top_n(predictions, n=10)
    
    
    # Print the recommended items for each user
    #for uid, user_ratings in top_n.items():
       # print(uid, [iid for (iid, _) in user_ratings])
    
    
    #tmp = pd.DataFrame(top_n[1]).rename(columns={0: 'MovieID', 1: 'Rating'}).merge(movies, on='MovieID')
    #print(tmp[['Title', 'Rating']])
    return pd.DataFrame(top_n[user_id]).rename(columns={0: 'MovieID', 1: 'Rating'})['MovieID'].tolist()


try:
    user_ratings=json.loads(sys.argv[1])
    
    e = create_engine("sqlite:///../moviereviews.db")
    movies = pd.DataFrame(e.execute("SELECT MovieID, Title, Genre FROM movies").fetchall()).rename(columns={0: 'MovieID', 1: 'Title', 2:'Genres'})
    ratings = pd.DataFrame(e.execute("SELECT UserID,MovieID,Rating FROM ratings").fetchall()).rename(columns={0: 'UserID', 1: 'MovieID', 2:'Rating'})
    
    
    
    #user_ratings = json.loads('{"110":5,"260":5,"1196":5,"1210":4,"1240":5,"2571":5,"3578":5}')
    user_ratings = {'UserID': [9998] * len(user_ratings.keys()),
                    'MovieID': list(user_ratings.keys()),
                    'Rating': list(user_ratings.values())}
    user_ratings = pd.DataFrame(user_ratings)
    
    ratings=pd.concat([user_ratings, ratings])
    
    
    t = time.time()
    return_json={}
    return_json['SVD'] = cf_recommend(ratings, SVD(reg_all=0.05, lr_all=0.007, n_epochs=30), 9998)
    return_json['COCLUSTERING'] = cf_recommend(ratings, CoClustering(), 9998)
    elapsed = time.time() - t
    
    for idx, movieID in enumerate(return_json['SVD']):
        return_json['SVD'][idx]={'MovieID':movieID, 'Title': movies[movies['MovieID']==movieID]['Title'].to_list()[0]}
        
    for idx, movieID in enumerate(return_json['COCLUSTERING']):
        return_json['COCLUSTERING'][idx]={'MovieID':movieID, 'Title': movies[movies['MovieID']==movieID]['Title'].to_list()[0]}
        
    
    print(json.dumps(return_json))
except:
    print("An exception in python occurred")




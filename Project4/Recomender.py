import pandas as pd
import numpy as np
import datetime
import time
from collections import defaultdict
from surprise import SVD, SlopeOne
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV


#dataframe headers
user_header = ["UserID","Gender","Age","Occupation","Zip-code"]
ratings_header = ['UserID','MovieID','Rating','Timestamp']
movies_header = ['MovieID','Title','Genres']

#import data
users = pd.read_csv('MovieData/users.dat',sep='::',names=user_header)
ratings = pd.read_csv('MovieData/ratings.dat',sep='::',names=ratings_header)
movies = pd.read_csv('MovieData/movies.dat',sep='::',names=movies_header)

ratings['datetime'] = ratings['Timestamp'].apply(datetime.datetime.fromtimestamp)

firsttime = min(ratings['Timestamp'])
lasttime = max(ratings['Timestamp'])

def time_to_coef(timestamp):
    return (timestamp-firsttime)/(lasttime-firsttime)

ratings['time_coef'] = ratings['Timestamp'].apply(time_to_coef)

reader = Reader()
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)
                                   
                                       
#split movie genres
movies_genres = pd.concat([pd.Series(row['MovieID'], row['Genres'].split('|')) for _, row in movies.iterrows()]).reset_index().rename(columns={'index': 'genre', 0: 'MovieID'})



avg_time_coef = ratings.groupby(by=['MovieID'])['time_coef'].agg([np.mean]).rename(columns={'mean': 'avg_time_coef'})
movie_grouped = ratings.groupby(by=['MovieID'])['Rating'].agg([np.sum, np.mean, np.std,np.ma.count]).merge(avg_time_coef, on='MovieID')


#tmp = movie_grouped.merge(avg_time_coef, on='MovieID')




ninty_pct = np.percentile(movie_grouped['count'],90)
fifty_pct = np.percentile(movie_grouped['count'],50)

meanreview = np.mean(ratings['Rating'])

#calculating weighted review
movie_grouped['wr'] = (movie_grouped['mean']*movie_grouped['count']/(movie_grouped['count']+ninty_pct)) + (meanreview*ninty_pct/(movie_grouped['count']+ninty_pct))

#calculating trendy reviews
movie_grouped['tr'] = movie_grouped['avg_time_coef']*((movie_grouped['mean']*movie_grouped['count']/(movie_grouped['count']+fifty_pct)) + (meanreview*fifty_pct/(movie_grouped['count']+fifty_pct)))



movies = movies.set_index('MovieID')

#tmp = movie_grouped.sort_values(by=['wr'], ascending=False).head(10).join(movies)
#movie_grouped['MovieID']=movie_grouped.index

genre_rating = movie_grouped.merge(movies_genres, on='MovieID')
unique_genres = movies_genres['genre'].unique()

def best_of_genre_1(g='Animation',n=10):
    genre_filtered = genre_rating[genre_rating['genre']==g]
    return genre_filtered.sort_values(by=['wr'], ascending=False).head(n)['MovieID'].tolist()


def best_of_genre_2(g='Animation',n=10):
    genre_filtered = genre_rating[genre_rating['genre']==g]
    return genre_filtered.sort_values(by=['tr'], ascending=False).head(n)['MovieID'].tolist()





for i in best_of_genre_1():
    print(movies.loc[i])


for i in best_of_genre_2():
    print(movies.loc[i])


# Load the movielens-100k dataset (download it if needed).

#data = Dataset.load_builtin('ml-1m', prompt = False)
# Use the famous SVD algorithm.


# Run 5-fold cross-validation and print results.
#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
#algo = KNNBasic()
# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE'], cv=5,  verbose=True);


#param_grid = {'n_epochs': [30], 'lr_all': [0.007],'reg_all': [0.05]}


#gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)

#gs.fit(data)

# best RMSE score
#print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
#print(gs.best_params['rmse'])




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



t = time.time()
elapsed = time.time() - t

def cf_recommend(data, algo, user_id):
    # First train an SVD algorithm on the movielens dataset.
    #data = Dataset.load_builtin('ml-100k')
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
    
    
    tmp = pd.DataFrame(top_n[1]).rename(columns={0: 'MovieID', 1: 'Rating'}).merge(movies, on='MovieID')
    #print(tmp[['Title', 'Rating']])
    return list(tmp['MovieID'])




recommended_movies_svd = cf_recommend(data, SVD(reg_all=0.05, lr_all=0.007, n_epochs=30), 1)


recommended_movies_slope1 = cf_recommend(data, SlopeOne(), 1)





























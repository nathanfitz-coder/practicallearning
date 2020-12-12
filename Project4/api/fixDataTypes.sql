ALTER TABLE movies rename to moviesold;
ALTER TABLE ratings rename to ratingsold;
ALTER TABLE users rename to usersold;


CREATE TABLE movies (
	MovieID integer,
   	Title TEXT,
	Genre TEXT,
	WeightedRating REAL,
	TrendRating REAL
);

CREATE TABLE ratings (
	MovieID INTEGER,
   	UserID INTEGER,
	Rating INTEGER,
	[Timestamp] INTEGER
);

CREATE TABLE users (
   	UserID INTEGER,
	Gender TEXT,
	Age INTEGER,
	Occupation INTEGER,
	Zipcode TEXT
);


INSERT INTO movies(
MovieID,
Title,
Genre
)
SELECT CAST(MovieID AS INTEGER) MovieID,
Title,
Genre
FROM moviesold;

INSERT INTO ratings(
MovieID,
UserID,
Rating,
[Timestamp]
)
SELECT CAST(MovieID AS INTEGER) MovieID,
CAST(UserID AS INTEGER) UserID,
CAST(Rating AS INTEGER) Rating,
CAST([Timestamp] AS INTEGER) [Timestamp]
FROM ratingsold;

INSERT INTO users(
UserID,
Gender,
Age,
Occupation,
Zipcode
)
SELECT CAST(UserID AS INTEGER) UserID,
Gender,
CAST(Age AS INTEGER) Age,
CAST(Occupation AS INTEGER) Occupation,
Zipcode
FROM usersold;

DROP TABLE moviesold;
DROP TABLE usersold;
DROP TABLE ratingsold;



CREATE TABLE moviesnew (
	MovieID integer,
   	Title TEXT,
	Genre TEXT,
	WeightedRating REAL,
	TrendRating REAL
);


INSERT INTO moviewsnew(
	MovieID ,
  Title ,
	Genre ,
	WeightedRating ,
	TrendRating
)

SELECT movies.MovieID
,movies.Title
,movies.Genre
,movie_weighted.weighted_review as WeightedRating
,movie_weighted.trendy_review as TrendRating
FROM movies
INNER JOIN movie_weighted
ON movies.MovieID = movie_weighted.MovieID;

DROP TABLE movies;

ALTER TABLE moviesnew rename to movies;

DROP TABLE movie_weighted;

Select Title
,movie_genres.Genre
,WeightedRating
FROM movies
INNER JOIN movie_genres
ON movies.MovieID = movie_genres.MovieID
WHERE movie_genres.Genre = 'Action'
ORDER BY WeightedRating desc;

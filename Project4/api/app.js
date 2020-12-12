//jshint esversion:6
//conda run -n tensorflow python ../Recomender.py
//start Server
//nodemon app.js

const express = require("express");
var cors = require('cors');
const {spawn} = require('child_process');
const bodyParser = require("body-parser");

//npm install sqlite@3

const sqlite3 = require('sqlite3').verbose();
var db = new sqlite3.Database('moviereviews.db');


const app = express();

//app.set('view engine', 'ejs');

app.use(bodyParser.urlencoded({extended: true}));
app.use(bodyParser.json());
app.use(express.static(__dirname));
app.use(cors());

app.get('/', function(req, res){
  res.sendFile(__dirname +"/index.html")
});

app.get("/genre", function(req, res){
  res.set('Access-Control-Allow-Origin', '*');
  let genreq = req.query.genreq;

  var query = `SELECT MovieID
  ,Title
  ,Type
  FROM (
  Select
  movies.MovieID
  ,Title
  ,'Weighted' Type
  ,movies.WeightedRating
  FROM movies
  INNER JOIN movie_genres
  ON movies.MovieID = movie_genres.MovieID
  WHERE movie_genres.Genre = '` + genreq + `'
  ORDER BY movies.WeightedRating desc
  LIMIT 20) a

  UNION ALL

  SELECT MovieID
  ,Title
  ,Type
  FROM (
  Select
  movies.MovieID
  ,Title
  ,'Trend' Type
  ,movies.TrendRating
  FROM movies
  INNER JOIN movie_genres
  ON movies.MovieID = movie_genres.MovieID
  WHERE movie_genres.Genre = '` + genreq + `'
  ORDER BY movies.TrendRating desc
  LIMIT 20) b`;


  db.all(query, function(err, rows) {
    if (err == null){
      res.send(JSON.stringify(rows));
    }else{
      console.log(err);
    }
   });
});


app.get("/cf", function(req, res){
  res.set('Access-Control-Allow-Origin', '*');
  let genreq = req.query.genreq;

  var query = `SELECT MovieID
  ,Title
  ,Genre
  FROM (
  Select
  movies.MovieID
  ,Title
  ,movie_genres.Genre
  ,movies.WeightedRating
  FROM movies
  INNER JOIN movie_genres
  ON movies.MovieID = movie_genres.MovieID
  WHERE movie_genres.Genre = '` + genreq + `'
  ORDER BY WeightedRating desc
  LIMIT 40) a`;


  db.all(query, function(err, rows) {
    if (err == null){
      res.send(JSON.stringify(rows));
    }else{
      console.log(err);
    }
   });
});

app.post("/ratings", function(req, res){
  //res.set('Access-Control-Allow-Origin', '*');
  var dataToSend;
  // spawn new child process to call the python script
  //const python = spawn('conda', ['run', '-n', 'tensorflow', 'python', 'Recomender.py', JSON.stringify(req.body)]);
  const python = spawn('python', ['Recomender.py', JSON.stringify(req.body)]);
  // collect data from script
  python.stdout.on('data', function (data) {
   console.log('Pipe data from python script ...');
   dataToSend = data.toString();
   console.log(dataToSend);

   res.send(dataToSend);



  });
  // in close event we are sure that stream from child process is closed
  python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`);
    // send data to browser
    //res.send(dataToSend)
  });

  console.log('Got body:', req.body);
  //res.sendStatus(200);
});


app.listen(process.env.PORT || 3000, function() {
  console.log("Server started on port 3000");
});

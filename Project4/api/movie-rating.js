var uniqueMovies = [2858,1617];
var movieData = {};
var movieRatings = {};
var ratingMovie = 0;
var totaltime = 0;
var timeinc = 5000;


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


async function tryGetRecommendations(url="/recready"){
    var res = await fetch(url)
    .then(
      function(response) {
        if (response.status == 202){
          console.log('Recommendations are not ready yet: ' + response.status);
          waitForRecs();
          return response.status
        } else if (response.status == 200){
          console.log('Recommendations ready: ' + response.status);
          response.json().then(function(data) {
            showRecommendations(data);
          });
          return response.status;
        }
        else {
          console.log('Looks like there was a problem. Status Code: ' + response.status);
          return response.status;
        }
      }
    ).catch(function(err) {
      console.log('Fetch Error :-S', err);
    });
    return res.text;
}


function waitForRecs(){
  if(totaltime < 300000){
    sleep(5000).then(() => { tryGetRecommendations(url="/recready"); });
    totaltime = totaltime + timeinc;
  }


}





function getMovieData(url){
  var res = fetch(url)
  .then(
    function(response) {
      if (response.status !== 200) {
        console.log('Looks like there was a problem. Status Code: ' +
          response.status);
        return;
      }
      // Examine the text in the response
      response.json().then(function(data) {
        insertMovies(data);
      });
    }
  )
  .catch(function(err) {
    console.log('Fetch Error :-S', err);
  });
}


//body: JSON.stringify(postData)
//body: "{'100': '5'}"
//JSON.stringify(data).length.toString()
async function postData(url = '', data = {}) {
  // Default options are marked with *
  const response = await fetch(url, {
    method: 'POST', // *GET, POST, PUT, DELETE, etc.
    mode: 'cors', // no-cors, *cors, same-origin
    cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
    credentials: 'same-origin', // include, *same-origin, omit
    headers: {
      'Content-Type': 'application/json'
    },
    redirect: 'follow', // manual, *follow, error
    referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
    body: JSON.stringify(data) // body data type must match "Content-Type" header
  });

  //return response   //.json(); // parses JSON response into native JavaScript objects
}



function pleaseWait(){
  $("h1").html("Your Recommendations Are Being Prepared");
  $(".button_caption").html("<p>This may take a few minutes...</p>");
  $(".all_movies").empty();
}

function showRecommendations(data){
  totaltime = 0;
  $("h1").html("I Think You'll Like These!");
  $(".button_caption").html("<p>The name of the column represents the algorithm used...</p>");
  $(".all_movies").empty();

  $(".all_movies").append("<div class='genre_column' id='SVD'><h3>SVD</h3></div>");
  $.each(data.SVD, function( index, value ) {
    $("#SVD").append("<div class='movie'><img src='https://github.com/liangfgithub/liangfgithub.github.io/blob/master/MovieImages/" + value.MovieID + ".jpg?raw=True alt='Not Available'><p class='movieTitle'>" + value.Title + "</p></div>");
  });

  $(".all_movies").append("<div class='genre_column' id='COCLUSTERING'><h3>Co-Clustering</h3></div>");
  $.each(data.COCLUSTERING, function( index, value ) {
    $("#COCLUSTERING").append("<div class='movie'><img src='https://github.com/liangfgithub/liangfgithub.github.io/blob/master/MovieImages/" + value.MovieID + ".jpg?raw=True alt='Not Available'><p class='movieTitle'>" + value.Title + "</p></div>");
  });

}


function insertMovies(data){

  var initCol = "<div class='genre_column' id='" + data[0].Genre + "'><h3>" + data[0].Genre + "</h3></div>"
  //alert(initCol);
  $(".all_movies").append(initCol);
  var cnt = 0;
  $.each(data, function( index, value ) {
    if(!uniqueMovies.includes(value.MovieID) && cnt < 20){
      $("#" + data[0].Genre).append("<div class='cf movie' title='"+value.MovieID+"' data-movieid='" + value.MovieID + "' onclick='clickMovie(" + value.MovieID + ")'><img src='https://github.com/liangfgithub/liangfgithub.github.io/blob/master/MovieImages/" + value.MovieID + ".jpg?raw=True alt='Not Available'><div class='moviestars'></div><p class='movieTitle'>" + value.Title + "</p></div>");
      uniqueMovies.push(value.MovieID);
      movieData[value.MovieID]= value.Title;
      cnt++;
    }
  });
}

function clickMovie(movieID){
  ratingMovie = movieID;
  $("#ratingDiv").removeClass("hidden");
  $("#ratingDiv > h4").text(movieData[movieID]);
}

function rateMovie(stars){
  $("#ratingDiv").addClass("hidden");
  movieRatings[ratingMovie]=stars;
  $(".moviestars",$("div").find(`[data-movieid='${ratingMovie}']`)).empty();
  for (i = 0; i < stars; i++) {
    $(".moviestars",$("div").find(`[data-movieid='${ratingMovie}']`)).append("<svg class='starthumb' enable-background='new 0 0 24 24' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='m23.363 8.584-7.378-1.127-3.307-7.044c-.247-.526-1.11-.526-1.357 0l-3.306 7.044-7.378 1.127c-.606.093-.848.83-.423 1.265l5.36 5.494-1.267 7.767c-.101.617.558 1.08 1.103.777l6.59-3.642 6.59 3.643c.54.3 1.205-.154 1.103-.777l-1.267-7.767 5.36-5.494c.425-.436.182-1.173-.423-1.266z' /></svg>");
  }
}


$.each(["Sci-Fi","Drama","Action","Animation","Childrens","Crime"], function( index, value ) {
  var url = '/cf?genreq=' + value;
  getMovieData(url);
});

$("#cf_submit_bttn").click(function(){
  // alert('sending post: ' + JSON.stringify(movieRatings));
  pleaseWait();
  postData('/ratings',movieRatings);

  //  .then(data => {
  //    showRecommendations(data);
  //    console.log(data);
  //});
  tryGetRecommendations(url="/recready");
});

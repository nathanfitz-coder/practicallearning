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



function insertMovies(data){
  weightedData = $.grep( data, function( n, i ) {
    return n.Type==='Weighted';
  });

  trendData = $.grep( data, function( n, i ) {
    return n.Type==='Trend';
  });

  if($("#weighted").children().length==0){
    $("#weighted").append("<h3>Top Movies</h3>");
    $.each(weightedData, function( index, value ) {
      $("#weighted").append("<div class='movie'><img src='' alt='Not Available'><p class='movieTitle'></p></div>");
    });
  }
  if($("#trend").children().length==0){
    $("#trend").append("<h3>Trending Movies</h3>");
    $.each(trendData, function( index, value ) {
      $("#trend").append("<div class='movie'><img src='' alt='Not Available'><p class='movieTitle'></p></div>");
    });
  }
  $.each(weightedData, function( index, value ) {
    var currentMovie = $("#weighted > .movie").eq(index);
    $('> img', currentMovie).attr("src","https://github.com/liangfgithub/liangfgithub.github.io/blob/master/MovieImages/" + value.MovieID + ".jpg?raw=True");
    $('> p', currentMovie).html(value.Title);
  });
  $.each(trendData, function( index, value ) {
    var currentMovie = $("#trend > .movie").eq(index);
    $('> img', currentMovie).attr("src","https://github.com/liangfgithub/liangfgithub.github.io/blob/master/MovieImages/" + value.MovieID + ".jpg?raw=True");
    $('> p', currentMovie).html(value.Title);
  });
  //$("#weighted").append("<div class='movie'><img src='https://github.com/liangfgithub/liangfgithub.github.io/blob/master/MovieImages/" + data[1].MovieID + ".jpg?raw=True' alt='Notta'><p class='movieTitle'>" + data[1].Title + "</p></div>");
  //$("#weighted").append("<div class='movie'><img src='https://github.com/liangfgithub/liangfgithub.github.io/blob/master/MovieImages/" + data[2].MovieID + ".jpg?raw=True' alt='Notta'><p class='movieTitle'>" + data[2].Title + "</p></div>");
  //alert($("#weighted").length);
}


// var myOptions = {
//     val1 : 'Blue',
//     val2 : 'Orange'
// };
// var mySelect = $('#myColors');
// $.each(myOptions, function(val, text) {
//     mySelect.append(
//         $('<option></option>').val(val).html(text)
//     );
// });

$('#genreSelect').click(function(){
  var url = '/genre?genreq=' + $(this).val()
  getMovieData(url);
});

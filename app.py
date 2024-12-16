import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ipywidgets import interact, widgets
import io
import base64
from wordcloud import WordCloud
import plotly.express as px


movie_final = pd.read_csv('_data/movie_final.csv')

LDA_df = pd.read_csv('_data/LDA_movie.csv')
all_genres = LDA_df['Movie genres'].explode()
genres_counts = all_genres.value_counts()


def str_to_array(str_vec):
    # Strip brackets and spaces, then split the string by whitespace
    return np.array([float(x) for x in str_vec.strip('[]').split()], dtype=np.float64)

# Apply the conversion to the 'Topics' column
movie_final['Topics'] = movie_final['Topics'].apply(str_to_array)

def recommend_similar_movie(movie_name):
  # Check that movie name is in the dataframe
  if not (movie_final['Movie name'] == movie_name).any():
    return "Error: This Movie is either mispelled or not in the database."

  else:
    # From movie name get topic score
    movie_infos = movie_final[movie_final['Movie name'] == movie_name].iloc[0]
    other_movies = movie_final[~(movie_final['Movie name'] == movie_name)]

    # Euclidean distances
    distances = other_movies['Topics'].apply(
        lambda x: np.linalg.norm(x- movie_infos['Topics'])
    )

    closest_movie = distances.nsmallest(1).index[-1]
    return other_movies['Movie name'].iloc[closest_movie]


app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['POST'])

def recommend():
    if request.is_json:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the movie name from the JSON data
        movie_name = data.get('movie_name', '')

        recommendation = recommend_similar_movie(movie_name)
        
        # Your recommendation logic here
        #recommendation = f"Recommended movie based on {movie_name}"
        
        return jsonify({'recommendation': recommendation})
    else:
        return jsonify({'error': 'Request must be JSON'}), 400



if __name__ == '__main__':
    app.run(debug=True)
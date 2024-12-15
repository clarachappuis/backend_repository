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

def generate_wordcloud_for_genre(genre):
    # Filter movies by the selected genre
    genre_movies = LDA_df[LDA_df['Movie genres'].apply(lambda x: genre in x)]
    
    # Combine the synopses into one string
    all_synopses = " ".join(genre_movies['plot_synopsis'].dropna())
    
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_synopses)
    
    # Save the word cloud to a BytesIO object
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    
    # Convert image to base64 to display in HTML
    img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return img_b64

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
    
  

@app.route('/')
def index():
    # Generate the word cloud for the selected genre (using a default genre for now)
    selected_genre = genres_counts.index[0]  # Default to the first genre
    wordcloud_img_b64 = generate_wordcloud_for_genre(selected_genre)

    # Create a Plotly bar chart to display genre counts
    fig = px.bar(
        x=genres_counts.index,
        y=genres_counts.values,
        labels={'x': 'Genre', 'y': 'Count'},
        title="Movie Genre Counts"
    )

    # Convert the Plotly figure to HTML for embedding in the template
    plot_html = fig.to_html(full_html=False)

    return render_template('index.html', plot=plot_html, wordcloud_img=wordcloud_img_b64)



if __name__ == '__main__':
    app.run(debug=True)
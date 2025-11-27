import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Movies%20Recommendation.csv"
    movies = pd.read_csv(url)
    return movies

@st.cache_resource
def build_model(movies):
    movies_features = movies[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline',
                              'Movie_Cast', 'Movie_Director']].fillna('')

    combined_features = (
        movies_features['Movie_Genre'] + ' ' +
        movies_features['Movie_Keywords'] + ' ' +
        movies_features['Movie_Tagline'] + ' ' +
        movies_features['Movie_Cast'] + ' ' +
        movies_features['Movie_Director']
    )

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(combined_features)
    similarity = cosine_similarity(matrix)

    return similarity

def recommend_movie(movie_name, movies, similarity_matrix, top_n=10):
    movie_list = movies['Movie_Title'].tolist()
    matches = difflib.get_close_matches(movie_name, movie_list)

    if not matches:
        return ["No similar movie found"]

    close_match = matches[0]
    movie_index = movies[movies.Movie_Title == close_match].index[0]

    scores = list(enumerate(similarity_matrix[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, movie in enumerate(sorted_scores[1: top_n + 1]):
        index = movie[0]
        movie_title = movies.iloc[index]['Movie_Title']
        recommendations.append(movie_title)

    return recommendations

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Built with **Cosine Similarity + TF-IDF**")

movies = load_data()
similarity_matrix = build_model(movies)

movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name!")
    else:
        results = recommend_movie(movie_name, movies, similarity_matrix, top_n=10)
        st.subheader("Top 10 Recommendations")
        for i, movie in enumerate(results, start=1):
            st.write(f"**{i}. {movie}**")

# movie_recommender_app.py

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Step 1: Load and preprocess data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df.dropna(subset=['genres'], inplace=True)
    df['genres'] = df['genres'].str.replace('|', ' ', regex=False)
    return df

movies_df = load_data()

# -----------------------------
# Step 2: TF-IDF Vectorization on genres
# -----------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

# -----------------------------
# Step 3: Compute cosine similarity matrix
# -----------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# Step 4: Recommendation function
# -----------------------------
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies_df[movies_df['title'] == title].index
    if idx.empty:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 excluding itself
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# -----------------------------
# Step 5: Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommender System")
st.write("Select a movie to get similar recommendations based on genres.")

movie_list = movies_df['title'].sort_values().tolist()
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        st.subheader("You might also like:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("No recommendations found.")

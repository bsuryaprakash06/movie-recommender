import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "96e50146e73e24e122673a8b5eae6e24"  

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# --- Apply custom style ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🎬 CineMatch: Your Movie Companion</h1>", unsafe_allow_html=True)


# --- Load movie data ---
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'overview']].dropna()
    return df

movies = load_data()

# --- Compute TF-IDF similarity ---
@st.cache_resource
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    return cosine_similarity(tfidf_matrix)

similarity = compute_similarity(movies)

# --- Poster Fetcher ---
def fetch_poster(movie_title):
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        res = requests.get(search_url).json()
        if not res['results']:
            return "https://via.placeholder.com/300x450?text=No+Image"

        movie_id = res['results'][0]['id']
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        details = requests.get(details_url).json()
        poster_path = details.get('poster_path')

        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Image"
    except:
        return "https://via.placeholder.com/300x450?text=No+Image"

# --- Recommendation Logic ---
def recommend(movie_name):
    movie_name = movie_name.lower()
    try:
        idx = movies[movies['title'].str.lower() == movie_name].index[0]
    except IndexError:
        return []

    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_titles = [movies.iloc[i[0]].title for i in sorted_scores]
    posters = [fetch_poster(title) for title in recommended_titles]
    return list(zip(recommended_titles, posters))

# --- User Input ---
movie_input = st.selectbox(
    "Search for a movie",
    sorted(movies['title'].tolist()),
    index=None,
    placeholder="Start typing..."
)

# --- Show Recommendations ---
if movie_input:
    with st.spinner("Fetching recommendations..."):
        results = recommend(movie_input)

    if not results:
        st.warning("Movie not found or no recommendations available.")
    else:
        st.subheader("🎯 Recommendations")
        cols = st.columns(len(results))
        for i, (title, poster_url) in enumerate(results):
            with cols[i]:
                st.image(poster_url, use_container_width=True)
                st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 13px;'>Built by Shane · Powered by Streamlit</div>", unsafe_allow_html=True)

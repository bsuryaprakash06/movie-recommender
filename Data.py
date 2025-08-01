import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv("C:/Users/admin/Documents/archive/tmdb_5000_movies.csv")

# Keep only the necessary columns
movies = movies[['id', 'title', 'overview']]
movies.dropna(inplace=True)

# Convert overview text into TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
vectors = tfidf.fit_transform(movies['overview'])

# Calculate cosine similarity between vectors
similarity = cosine_similarity(vectors)

# Recommend function
def recommend(movie_name):
    movie_name = movie_name.lower()
    idx = None

    for i, title in enumerate(movies['title']):
        if title.lower() == movie_name:
            idx = i
            break

    if idx is None:
        print("Movie not found.")
        return

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print(f"Movies similar to '{movies.iloc[idx].title}':")
    for i in scores:
        print(movies.iloc[i[0]].title)

# Test it
recommend("Avatar")


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('data/books.csv')

# Show basic info
print("Shape of dataset:", df.shape)

# Drop useless column
df = df.drop(columns=["Unnamed: 12"], errors='ignore')

# Fill NaNs with empty string
df.fillna('', inplace=True)

# Combine relevant text fields into a single "content" field
df['content'] = df['title'] + ' ' + df['authors'] + ' ' + df['publisher']

# Vectorize using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Build a mapping of book titles to DataFrame indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommend books based on a given title
def recommend_books(title, num_recommendations=5):
    # Check if title is in the dataset
    if title not in indices:
        print(f"'{title}' not found in the dataset.")
        return

    idx = indices[title]

    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity score (excluding itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the titles
    print(f"\n📚 Recommendations for '{title}':")
    for i, book in enumerate(df['title'].iloc[book_indices], 1):
        print(f"{i}. {book}")

# Test the recommender
# Try replacing the title below with any book in the dataset
recommend_books('Harry Potter and the Chamber of Secrets (Harry Potter, #2)')
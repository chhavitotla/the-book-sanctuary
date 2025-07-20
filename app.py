import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="The Book Sanctuary",
    layout="centered",
    initial_sidebar_state="auto"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        background: linear-gradient(135deg, #F8E4E9, #F2D16B);
        font-family: 'Inter', sans-serif;
        color: #2D2D2D;
    }

    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif;
        color: #2D2D2D;
    }

    .stTextInput>div>div>input {
        background-color: #fff8f9;
        border: 1px solid #ccc;
        border-radius: 10px;
    }

    .stSlider > div {
        color: #2D2D2D;
    }

    .stButton>button {
        background-color: #B185A7;
        color: #FAF3E0;
        border: none;
        padding: 0.6em 1.5em;
        border-radius: 30px;
        font-weight: 600;
        transition: 0.3s ease;
        font-size: 1em;
    }

    .stButton>button:hover {
        background-color: #A2678A;
        box-shadow: 0 0 12px rgba(178, 133, 167, 0.5);
        color: #fff;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/books.csv')
    df = df.drop(columns=["Unnamed: 12"], errors='ignore')
    df.fillna('', inplace=True)
    df['content'] = df['title'] + ' ' + df['authors'] + ' ' + df['publisher']
    return df

df = load_data()

# -------------------- TF-IDF SETUP --------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
df['title_lower'] = df['title'].str.lower()

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align: center;'>The Book Sanctuary</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>This isn’t just a rec — it’s your next personality trait.</p>", unsafe_allow_html=True)
st.markdown("###")

# -------------------- INPUT --------------------
book_name = st.text_input("🔍 Enter a book title:", placeholder="e.g. Harry Potter and the Half-Blood Prince")
num_recs = st.slider("📝 Number of recommendations:", min_value=1, max_value=10, value=5)

# -------------------- RECOMMENDER --------------------
def recommend_books(title, num_recommendations=5):
    title = title.strip().lower()
    all_titles = df['title_lower'].tolist()

    close_matches = get_close_matches(title, all_titles, n=1, cutoff=0.6)

    if not close_matches:
        return []

    matched_title = close_matches[0]
    idx = df[df['title_lower'] == matched_title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices].tolist()

# -------------------- BUTTON + RESULTS --------------------
if st.button("✨ Get My Book Match"):
    if book_name.strip() == "":
        st.warning("Please enter a valid book title.")
    else:
        results = recommend_books(book_name, num_recs)
        if not results:
            st.error("Sorry, that book isn’t in our library. Try a different one!")
        else:
            st.markdown("### 💫 You might also like:")
            for book in results:
                st.markdown(f"- {book}")
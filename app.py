import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import time

# Page config
st.set_page_config(
    page_title="OnceUponAI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load resources
@st.cache_resource
def load_resources():
    index = faiss.read_index('books.index')
    df = pd.read_pickle('books.pkl')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, df, model

try:
    index, df, model = load_resources()
except:
    st.error("⚠️ Please run `python build_index.py` first to create the book index!")
    st.stop()

# Custom CSS for carousel and styling
st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main title styling */
    .main-title {
        font-size: 64px;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        padding-top: 20px;
    }
    
    .subtitle {
        text-align: center;
        color: #7F8C8D;
        font-size: 20px;
        margin-bottom: 40px;
    }
    
    /* Search box styling */
    .stTextArea textarea {
        font-size: 18px;
        border-radius: 15px;
        border: 2px solid #667eea;
    }
    
    /* Book card styling */
    .book-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px;
        transition: transform 0.3s;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    .book-title {
        font-size: 22px;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 5px;
    }
    
    .book-author {
        font-size: 18px;
        color: #7F8C8D;
        font-style: italic;
        margin-bottom: 10px;
    }
    
    .book-blurb {
        font-size: 16px;
        color: #34495E;
        line-height: 1.6;
    }
    
    .call-number {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
        font-size: 16px;
        margin-top: 10px;
    }
    
    /* Carousel styling */
    .carousel-container {
        overflow: hidden;
        position: relative;
        height: 400px;
        margin: 40px 0;
    }
    
    /* Reduce padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for carousel
if 'carousel_index' not in st.session_state:
    st.session_state.carousel_index = 0
    st.session_state.carousel_books = df.sample(n=min(20, len(df))).reset_index(drop=True)

# Auto-rotate carousel
if 'last_rotation' not in st.session_state:
    st.session_state.last_rotation = time.time()

# Rotate every 5 seconds
current_time = time.time()
if current_time - st.session_state.last_rotation > 5:
    st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(st.session_state.carousel_books)
    st.session_state.last_rotation = current_time
    st.rerun()

# Header
st.markdown("<h1 class='main-title'>✨ OnceUponAI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover your next great read with AI</p>", unsafe_allow_html=True)

# Search Section
st.markdown("### 🔍 Find Your Perfect Book")
query = st.text_area(
    "Describe what you're looking for:",
    placeholder="e.g., 'A thrilling mystery in Victorian London' or 'An inspiring story about overcoming challenges'",
    height=100,
    key="search_input"
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

if search_button and query.strip():
    with st.spinner("🔮 Searching through our collection..."):
        # Encode query
        query_embedding = model.encode([query])
        
        # Search FAISS
        distances, indices = index.search(query_embedding.astype('float32'), 5)
        
        st.markdown("---")
        st.markdown("### 📚 Best Matches")
        
        # Display results in a grid
        for i in range(0, len(indices[0]), 2):
            cols = st.columns(2)
            
            for col_idx, result_idx in enumerate(indices[0][i:i+2]):
                if result_idx >= len(df):
                    continue
                    
                book = df.iloc[result_idx]
                similarity_score = 1 / (1 + distances[0][i + col_idx])
                
                with cols[col_idx]:
                    # Book card
                    card_col1, card_col2 = st.columns([1, 2])
                    
                    with card_col1:
                        cover_path = f"data/covers/{book.get('cover_filename', '')}"
                        if os.path.exists(cover_path):
                            st.image(cover_path, use_column_width=True)
                        else:
                            st.image("https://via.placeholder.com/300x450?text=No+Cover", use_column_width=True)
                    
                    with card_col2:
                        st.markdown(f"<p class='book-title'>{book['title']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='book-author'>by {book['author']}</p>", unsafe_allow_html=True)
                        
                        # Blurb (truncated)
                        blurb = book['blurb']
                        if len(blurb) > 200:
                            blurb = blurb[:197] + "..."
                        st.markdown(f"<p class='book-blurb'>{blurb}</p>", unsafe_allow_html=True)
                        
                        # Call number
                        if 'call_number' in book and pd.notna(book['call_number']) and book['call_number']:
                            st.markdown(f"<span class='call-number'>📍 {book['call_number']}</span>", unsafe_allow_html=True)
                        
                        # Match percentage
                        st.progress(similarity_score, text=f"{similarity_score:.0%} match")
                    
                    st.markdown("<br>", unsafe_allow_html=True)

elif search_button:
    st.warning("⚠️ Please enter a description to search!")

# Carousel Section
st.markdown("---")
st.markdown("### 📖 Discover Books from Our Collection")

# Get current book from carousel
current_book = st.session_state.carousel_books.iloc[st.session_state.carousel_index]

# Display carousel book
carousel_col1, carousel_col2, carousel_col3 = st.columns([1, 2, 1])

with carousel_col2:
    inner_col1, inner_col2 = st.columns([1, 2])
    
    with inner_col1:
        cover_path = f"data/covers/{current_book.get('cover_filename', '')}"
        if os.path.exists(cover_path):
            st.image(cover_path, use_column_width=True)
        else:
            st.image("https://via.placeholder.com/300x450?text=No+Cover", use_column_width=True)
    
    with inner_col2:
        st.markdown(f"<p class='book-title'>{current_book['title']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='book-author'>by {current_book['author']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='book-blurb'>{current_book['blurb']}</p>", unsafe_allow_html=True)
        
        if 'call_number' in current_book and pd.notna(current_book['call_number']) and current_book['call_number']:
            st.markdown(f"<span class='call-number'>📍 {current_book['call_number']}</span>", unsafe_allow_html=True)

# Manual carousel controls
st.markdown("<br>", unsafe_allow_html=True)
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([2, 1, 1, 1, 2])

with nav_col2:
    if st.button("⬅️ Previous", use_container_width=True):
        st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(st.session_state.carousel_books)
        st.rerun()

with nav_col3:
    st.markdown(f"<div style='text-align: center; padding: 8px;'>{st.session_state.carousel_index + 1} / {len(st.session_state.carousel_books)}</div>", unsafe_allow_html=True)

with nav_col4:
    if st.button("Next ➡️", use_container_width=True):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(st.session_state.carousel_books)
        st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #7F8C8D;'>✨ Featuring {len(df)} books from our collection • Auto-rotating every 5 seconds</div>", unsafe_allow_html=True)
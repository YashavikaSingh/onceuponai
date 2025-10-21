import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import time

# Page config
# st.set_page_config(
#     page_title="OnceUponAI",
#     page_icon="✨",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

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

# Custom CSS
st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Main title styling */
    .main-title {
        font-size: 72px;
        font-weight: bold;
        color: #57068c !important;
        text-align: left;
        margin-bottom: 30px;
        margin-top: 20px;
        letter-spacing: 2px;
        padding: 20px;
    }
    
    /* Left side - Carousel */
    .carousel-book-title {
        font-size: 48px;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 15px;
        line-height: 1.2;
    }
    
    .carousel-book-author {
        font-size: 32px;
        color: #7F8C8D;
        font-style: italic;
        margin-bottom: 20px;
    }
    
    .carousel-book-blurb {
        font-size: 24px;
        color: #34495E;
        line-height: 1.8;
        margin-bottom: 20px;
        max-height: 500px;
        overflow-y: auto;
        padding-right: 10px;
    }
    
    /* Scrollbar styling */
    .carousel-book-blurb::-webkit-scrollbar {
        width: 8px;
    }
    
    .carousel-book-blurb::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .carousel-book-blurb::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .carousel-book-blurb::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    .carousel-call-number {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 10px;
        font-weight: bold;
        display: inline-block;
        font-size: 28px;
    }
    
    /* Right side - Search */
    .search-title {
        font-size: 36px;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 20px;
    }
    
    .stTextArea textarea {
        font-size: 20px;
        border-radius: 15px;
        border: 2px solid #667eea;
        min-height: 120px;
    }
    
    /* Search results */
    .result-book-title {
        font-size: 20px;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 5px;
    }
    
    .result-book-author {
        font-size: 16px;
        color: #7F8C8D;
        font-style: italic;
        margin-bottom: 8px;
    }
    
    .result-book-blurb {
        font-size: 14px;
        color: #34495E;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    
    .result-call-number {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
        font-size: 14px;
    }
    
    /* Divider */
    .vertical-divider {
        border-left: 3px solid #E0E0E0;
        height: 100vh;
        margin: 0 20px;
    }
        /* Reduce space between image and text columns */
    div[data-testid="column"] {
        padding-right: 0.25rem !important;
        padding-left: 0.25rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for carousel
if 'carousel_index' not in st.session_state:
    st.session_state.carousel_index = 0
    st.session_state.carousel_books = df.sample(n=min(50, len(df))).reset_index(drop=True)

if 'last_rotation' not in st.session_state:
    st.session_state.last_rotation = time.time()

if time.time() - st.session_state.last_rotation > 15:
    st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(st.session_state.carousel_books)
    st.session_state.last_rotation = time.time()
    st.experimental_rerun()

# No title at top (reduce top padding)
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# Two column layout - LEFT: Image | MIDDLE: Blurb | RIGHT: Search
left_col, middle_col, right_col = st.columns([1, 1, 1], gap="large")

# ========== LEFT SIDE: BOOK COVER ==========
with left_col:
    current_book = st.session_state.carousel_books.iloc[st.session_state.carousel_index]
    
    # Book cover
    cover_path = f"data/covers/{current_book.get('cover_filename', '')}"
    if os.path.exists(cover_path):
        st.image(cover_path, width=400)
    else:
        st.image("https://via.placeholder.com/400x600?text=No+Cover", width=400)

# ========== MIDDLE: BOOK DETAILS ==========
with middle_col:
    current_book = st.session_state.carousel_books.iloc[st.session_state.carousel_index]
    
    # Book details
    st.markdown(f"<p class='carousel-book-title'>{current_book['title']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='carousel-book-author'>by {current_book['author']}</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='carousel-book-blurb'>{current_book['blurb']}</div>", unsafe_allow_html=True)
    
    # Call number - using div instead of span for better display
    if 'call_number' in current_book and pd.notna(current_book['call_number']) and current_book['call_number'] != '':
        st.markdown(f"<div class='carousel-call-number'>📍 Find at: {current_book['call_number']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #999; font-size: 18px;'>No call number available</div>", unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("<br><br>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    with nav_col1:
        if st.button("⬅️ Previous", key="prev"):
            st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(st.session_state.carousel_books)
            st.session_state.last_rotation = time.time()  # Reset timer
            st.rerun()
    
    with nav_col2:
        st.markdown(f"<div style='text-align: center; padding: 8px; font-size: 18px;'>{st.session_state.carousel_index + 1} / {len(st.session_state.carousel_books)}</div>", unsafe_allow_html=True)
    
    with nav_col3:
        if st.button("Next ➡️", key="next"):
            st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(st.session_state.carousel_books)
            st.session_state.last_rotation = time.time()  # Reset timer
            st.rerun()
    
    st.markdown("<div style='text-align: center; color: #7F8C8D; margin-top: 20px; font-size: 16px;'>Auto-rotating every 15 seconds</div>", unsafe_allow_html=True)

# ========== RIGHT SIDE: SEARCH ==========
with right_col:
    st.markdown("<p class='search-title'>🔍 Find Your Perfect Book</p>", unsafe_allow_html=True)
    
    query = st.text_area(
        "Describe what you're looking for:",
        placeholder="e.g., 'A thrilling mystery in Victorian London' or 'An inspiring story about overcoming challenges'",
        height=120,
        key="search_input",
        label_visibility="collapsed"
    )
    
    search_button = st.button("🔍 Search", type="primary")
    
    if search_button and query.strip():
        with st.spinner("🔮 Searching through our collection..."):
            # Encode query
            query_embedding = model.encode([query])

            # Search FAISS
            distances, indices = index.search(query_embedding.astype('float32'), 5)
            results = [df.iloc[i] for i in indices[0] if i < len(df)]

            # Store search results in session state
            st.session_state.search_results = results
            st.session_state.search_index = 0  # start with first result
            st.session_state.search_time = time.time()
            st.rerun()

    elif "search_results" in st.session_state and st.session_state.search_results:
        results = st.session_state.search_results
        current_idx = st.session_state.search_index
        book = results[current_idx]

        # Display single book result cleanly
        result_col1, result_col2 = st.columns([1, 1.2])
        with result_col1:
            cover_path = f"data/covers/{book.get('cover_filename', '')}"
            if os.path.exists(cover_path):
                st.image(cover_path, use_column_width=True)
            else:
                st.image("https://via.placeholder.com/400x600?text=No+Cover", use_column_width=True)

        with result_col2:
            st.markdown(f"<p class='result-book-title' style='font-size:28px;'>{book['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='result-book-author' style='font-size:22px;'>by {book['author']}</p>", unsafe_allow_html=True)
            
            blurb = book['blurb']
            if len(blurb) > 500:
                blurb = blurb[:497] + "..."
            st.markdown(f"<p class='result-book-blurb' style='font-size:18px; line-height:1.6;'>{blurb}</p>", unsafe_allow_html=True)

            if 'call_number' in book and pd.notna(book['call_number']) and book['call_number']:
                st.markdown(f"<div class='result-call-number' style='font-size:16px;'>📍 {book['call_number']}</div>", unsafe_allow_html=True)
        
        # Navigation buttons for horizontal swipe effect
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        with nav_col1:
            if st.button("⬅️ Previous Book", key="search_prev") and current_idx > 0:
                st.session_state.search_index -= 1
                st.rerun()
        with nav_col2:
            st.markdown(f"<div style='text-align:center; font-size:16px; color:#7F8C8D;'>{current_idx+1} / {len(results)}</div>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("Next Book ➡️", key="search_next") and current_idx < len(results) - 1:
                st.session_state.search_index += 1
                st.rerun()

    
    elif search_button:
        st.warning("⚠️ Please enter a description to search!")
    else:
        # Show instructions when no search
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("💡 **How to search:**\n\n"
                "Describe the kind of book you're in the mood for. For example:\n\n"
                "- 'A fast-paced thriller with plot twists'\n"
                "- 'A heartwarming story about friendship'\n"
                "- 'Historical fiction set during World War II'\n"
                "- 'Something that will make me laugh'\n\n"
                "Our AI will find the perfect match!")

            

# Footer
st.markdown(f"<hr><div style='text-align: center; font-size: 20px; color: #7F8C8D;'><span style='font-size: 32px; font-weight: bold; color: #57068c;'>OnceUponAI ✨</span>&nbsp;&nbsp;|&nbsp;&nbsp;✨ Featuring {len(df)} books from our collection</div>", unsafe_allow_html=True)

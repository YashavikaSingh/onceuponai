import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

print("="*60)
print("🔨 OnceUponAI - Building Vector Index")
print("="*60)

print("\n📖 Loading data...")
try:
    df = pd.read_csv('data/books.csv')
    print(f"✅ Found {len(df)} books")
except FileNotFoundError:
    print("❌ Error: data/books.csv not found!")
    print("   Please run bulk_fetch_books.py first")
    exit(1)

print("\n🤖 Loading AI model...")
print("   (This may take a minute on first run - downloading model)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

print("\n🧠 Creating embeddings...")
print("   This combines title, author, and blurb for better search results")

# Combine title, author, and blurb for better matching
texts = []
for _, row in df.iterrows():
    text = f"{row['title']} by {row['author']}. {row['blurb']}"
    texts.append(text)

embeddings = model.encode(texts, show_progress_bar=True)
print(f"✅ Created embeddings for {len(embeddings)} books")

print("\n📊 Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))
print(f"✅ Index built with {index.ntotal} vectors")

print("\n💾 Saving index and data...")
faiss.write_index(index, 'books.index')
df.to_pickle('books.pkl')
print("✅ Saved to books.index and books.pkl")

print("\n" + "="*60)
print("🎉 SUCCESS!")
print("="*60)
print(f"Vector database ready with {index.ntotal} books")
print("\nNext step: Run your app!")
print("   streamlit run app.py")
print("="*60)
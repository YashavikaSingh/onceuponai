"""
Bulk fetch book data for large collections
Searches by title (and optionally author), fetches everything automatically
Usage: python bulk_fetch_books.py
"""

import requests
import pandas as pd
import time
from PIL import Image
from io import BytesIO
import os
from pathlib import Path

print("="*60)
print("📚 OnceUponAI - Bulk Book Data Fetcher")
print("="*60)

# Read your book list
print("\n📖 Reading your book list...")

# Try to read the file (supports CSV, Excel)
input_file = 'Leisure_Title_Author_Callnumber.xlsx'  # Change this to your filename

try:
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df_input = pd.read_excel(input_file)
    else:
        # Read CSV and handle different column name formats
        df_input = pd.read_csv(input_file)
    
    print(f"✅ Found {len(df_input)} books in {input_file}")
    
    # Normalize column names (remove spaces, lowercase)
    df_input.columns = df_input.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Map common column name variations
    column_mapping = {
        'call no': 'call_number'
    }
    df_input.rename(columns=column_mapping, inplace=True)
    
    print(f"   Columns found: {', '.join(df_input.columns)}")
    
except FileNotFoundError:
    print(f"❌ Could not find '{input_file}'")
    print("\nPlease create a file named 'book_list.csv' with your books.")
    print("Format: author, call no, title")
    exit(1)

# Check required columns
if 'title' not in df_input.columns:
    print("❌ Error: 'title' column not found!")
    print(f"   Found columns: {', '.join(df_input.columns)}")
    exit(1)

has_author = 'author' in df_input.columns
has_call_number = 'call_number' in df_input.columns

print(f"   ✓ Title column: Found")
print(f"   ✓ Author column: {'Found' if has_author else 'Not found'}")
print(f"   ✓ Call number column: {'Found' if has_call_number else 'Not found'}")

# Create directories
os.makedirs('data/covers', exist_ok=True)

books = []
failed = []
total = len(df_input)

print(f"\n🚀 Starting to fetch {total} books...")
print("   This will take approximately {:.0f} minutes\n".format(total * 1.5 / 60))

for i, row in df_input.iterrows():
    idx = i + 1
    title = str(row['title']).strip()
    author = str(row.get('author', '')).strip() if has_author else ''
    call_number = str(row.get('call_number', '')).strip() if has_call_number else ''
    
    # Skip empty rows
    if not title or title == 'nan':
        print(f"[{idx}/{total}] ⚠️  Skipping empty row")
        continue
    
    print(f"[{idx}/{total}] {title}" + (f" by {author}" if author and author != 'nan' else ""))
    
    try:
        # Build search query
        if author and author != 'nan':
            query = f"{title} {author}"
        else:
            query = title
        
        # Google Books API search
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {
            'q': query,
            'maxResults': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'items' not in data or len(data['items']) == 0:
            print(f"  ❌ Not found")
            failed.append({'title': title, 'author': author, 'call_number': call_number, 'reason': 'Not found'})
            time.sleep(1)
            continue
        
        book_data = data['items'][0]['volumeInfo']
        
        # Extract data
        found_title = book_data.get('title', title)
        found_authors = book_data.get('authors', [author] if author and author != 'nan' else ['Unknown Author'])
        found_author = ', '.join(found_authors)
        
        # Get ISBN
        isbn = ''
        if 'industryIdentifiers' in book_data:
            for identifier in book_data['industryIdentifiers']:
                if identifier['type'] in ['ISBN_13', 'ISBN_10']:
                    isbn = identifier['identifier']
                    break
        
        # Get description
        blurb = book_data.get('description', 'No description available.')
        blurb = blurb.replace('\n', ' ').replace('\r', ' ').strip()
        
        # Truncate if too long (keep it readable)
        if len(blurb) > 600:
            blurb = blurb[:597] + '...'
        
        # Get cover image
        cover_filename = ''
        image_links = book_data.get('imageLinks', {})
        cover_url = image_links.get('thumbnail') or image_links.get('smallThumbnail')
        
        if cover_url:
            # Remove zoom parameter for better quality
            cover_url = cover_url.replace('zoom=1', 'zoom=0')
            # Use HTTPS
            cover_url = cover_url.replace('http://', 'https://')
            
            try:
                img_response = requests.get(cover_url, timeout=10)
                img = Image.open(BytesIO(img_response.content))
                
                # Resize for consistency
                if img.width > 400:
                    ratio = 400 / img.width
                    new_size = (400, int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save with sanitized filename
                safe_title = "".join(c for c in found_title if c.isalnum() or c in (' ', '-', '_'))[:50]
                cover_filename = f"{safe_title}_{idx}.jpg"
                img.save(f"data/covers/{cover_filename}")
                print(f"  ✅ Saved with cover")
            except Exception as e:
                print(f"  ✅ Saved (no cover)")
        else:
            print(f"  ✅ Saved (no cover)")
        
        books.append({
            'title': found_title,
            'author': found_author,
            'blurb': blurb,
            'isbn': isbn,
            'call_number': call_number if call_number and call_number != 'nan' else '',
            'cover_filename': cover_filename
        })
        
        # Save progress every 50 books
        if idx % 50 == 0:
            temp_df = pd.DataFrame(books)
            temp_df.to_csv('data/books_progress.csv', index=False)
            print(f"\n  💾 Progress saved ({len(books)} books so far)\n")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        failed.append({'title': title, 'author': author, 'call_number': call_number, 'reason': str(e)})
    
    # Rate limiting - be nice to Google
    time.sleep(1.2)
    
    # Show progress every 10 books
    if idx % 10 == 0:
        print(f"\n--- Progress: {idx}/{total} ({idx/total*100:.1f}%) ---\n")

# Final save
print("\n" + "="*60)
print("💾 Saving final results...")

df_final = pd.DataFrame(books)
df_final.to_csv('data/books.csv', index=False)

# Save failed list
if failed:
    df_failed = pd.DataFrame(failed)
    df_failed.to_csv('data/books_failed.csv', index=False)

print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)
print(f"✅ Successfully fetched: {len(books)} books")
print(f"❌ Failed to fetch: {len(failed)} books")
print(f"📈 Success rate: {len(books)/total*100:.1f}%")

print(f"\n📄 Main data: data/books.csv")
print(f"🖼️  Covers: data/covers/ ({sum(1 for b in books if b['cover_filename'])} covers downloaded)")

if failed:
    print(f"⚠️  Failed books: data/books_failed.csv")
    print("   (You can manually add these later or retry)")

if has_call_number:
    books_with_call = sum(1 for b in books if b['call_number'])
    print(f"📍 Books with call numbers: {books_with_call}/{len(books)}")

print("\n" + "="*60)
print("🎉 DONE! Next steps:")
print("="*60)
print("1. Review data/books.csv")
print("2. Check data/books_failed.csv for any books that couldn't be found")
print("3. Run: python build_index.py")
print("4. Run: streamlit run app.py")
print("5. Enjoy OnceUponAI! ✨")
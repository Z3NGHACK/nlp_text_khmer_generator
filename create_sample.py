import wikipedia

wikipedia.set_lang('km')  # Set to Khmer

# List of Khmer Wikipedia article titles (add more if needed)
articles = ['ភ្នំពេញ', 'កម្ពុជា', 'អង្គរ', 'សៀមរាប', 'ខ្មែរ', 'ប្រាសាទអង្គរវត្ត']  # Example additions

corpus = ''

for title in articles:
    try:
        page = wikipedia.page(title)
        print(f"Successfully fetched: {title} ({len(page.content)} characters)")
        corpus += page.content + '\n\n'  # Add extra newline between articles
    except wikipedia.exceptions.PageError:
        print(f"Page not found: {title}")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation page for {title}: {e.options[:5]}... (skipping)")
    except Exception as e:
        print(f"Error fetching {title}: {e}")

# Save to file
with open('khmer_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(corpus)

print(f"\nCorpus saved to khmer_corpus.txt ({len(corpus)} characters total)")
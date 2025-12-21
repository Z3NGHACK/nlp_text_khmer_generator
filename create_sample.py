# import wikipedia

# wikipedia.set_lang('en')  # Set to English (or remove this line, as 'en' is the default)

# # List of English Wikipedia article titles (corresponding to your original Khmer ones)
# articles = ['Phnom Penh', 'Cambodia', 'Angkor', 'Siem Reap', 'Khmer people', 'Angkor Wat']  
# # You can add more English titles here, e.g., 'Khmer Empire', 'History of Cambodia', etc.

# corpus = ''

# for title in articles:
#     try:
#         page = wikipedia.page(title)
#         print(f"Successfully fetched: {title} ({len(page.content)} characters)")
#         corpus += page.content + '\n\n'  # Add extra newline between articles
#     except wikipedia.exceptions.PageError:
#         print(f"Page not found: {title}")
#     except wikipedia.exceptions.DisambiguationError as e:
#         print(f"Disambiguation page for {title}: {e.options[:5]}... (skipping)")
#     except Exception as e:
#         print(f"Error fetching {title}: {e}")

# # Save to file
# with open('english_corpus.txt', 'w', encoding='utf-8') as f:  # Changed filename for clarity
#     f.write(corpus)

# print(f"\nCorpus saved to english_corpus.txt ({len(corpus)} characters total)")




import wikipedia

# No need to set language; default is English ('en')

# List of English Wikipedia article titles (general knowledge topics for a diverse corpus)
# You can add more titles as needed for a larger corpus
articles = [
    'Earth', 'Universe', 'Life', 'Human', 'Mathematics', 'Science', 'Technology',
    'History', 'Art', 'Philosophy', 'Religion', 'Society', 'Language', 'Literature',
    'United States', 'China',
]

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
with open('english_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(corpus)

print(f"\nCorpus saved to english_corpus.txt ({len(corpus)} characters total)")

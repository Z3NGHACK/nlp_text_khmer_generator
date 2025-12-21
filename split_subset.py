import nltk
from sklearn.model_selection import train_test_split

# Download required NLTK data (only needed once)
nltk.download('punkt')  # Sentence tokenizer
nltk.download('punkt_tab')  # Additional for better performance

# Load corpus
with open('english_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Loaded English corpus: {len(text)} characters")

# Tokenize into sentences
sentences = nltk.sent_tokenize(text.strip())
print(f"Tokenized into {len(sentences)} sentences")

# Split: 70% train, 15% validation, 15% test
train_sentences, temp_sentences = train_test_split(sentences, test_size=0.30, random_state=42)
val_sentences, test_sentences = train_test_split(temp_sentences, test_size=0.50, random_state=42)

print(f"Train: {len(train_sentences)} sentences ({len(train_sentences)/len(sentences)*100:.1f}%)")
print(f"Validation: {len(val_sentences)} sentences ({len(val_sentences)/len(sentences)*100:.1f}%)")
print(f"Test: {len(test_sentences)} sentences ({len(test_sentences)/len(sentences)*100:.1f}%)")

# Save the splits
def save_sentences(sentences_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for sent in sentences_list:
            f.write(sent.strip() + '\n\n')  # Double newline for clear separation

save_sentences(train_sentences, 'train.txt')
save_sentences(val_sentences, 'val.txt')
save_sentences(test_sentences, 'test.txt')

print("English splits saved as train.txt, val.txt, test.txt")
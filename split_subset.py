from khmernltk import sentence_tokenize  # No need for word_tokenize here unless you use it later
from sklearn.model_selection import train_test_split
import random

# Load corpus
with open('khmer_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Loaded corpus: {len(text)} characters")

# Tokenize into sentences
sentences = sentence_tokenize(text)
print(f"Tokenized into {len(sentences)} sentences")

# Shuffle and split: 70% train, 10% val, 20% test
random.shuffle(sentences)
train_sentences, temp_sentences = train_test_split(sentences, test_size=0.3, random_state=42)
val_sentences, test_sentences = train_test_split(temp_sentences, test_size=2/3, random_state=42)

print(f"Train: {len(train_sentences)} sentences")
print(f"Validation: {len(val_sentences)} sentences")
print(f"Test: {len(test_sentences)} sentences")

# Save the splits (optional but useful)
def save_sentences(sentences_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for sent in sentences_list:
            f.write(sent + '\n\n')  # Double newline to separate sentences clearly

save_sentences(train_sentences, 'train.txt')
save_sentences(val_sentences, 'val.txt')
save_sentences(test_sentences, 'test.txt')

print("Splits saved as train.txt, val.txt, test.txt")
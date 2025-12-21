# main_project.py - ENGLISH VERSION (FINAL CORRECTED & OPTIMIZED)
import random
from collections import Counter, defaultdict
import math
import numpy as np
import nltk

# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==================== 1. Load the text files ====================
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

print("Loading English data...")
train_text = load_text('train.txt')
val_text = load_text('val.txt')
test_text = load_text('test.txt')

# ==================== 2. Sentence tokenization (English) ====================
print("Sentence tokenizing with NLTK...")
train_sents = nltk.sent_tokenize(train_text.strip())
val_sents = nltk.sent_tokenize(val_text.strip())
test_sents = nltk.sent_tokenize(test_text.strip())

print(f"Train sentences: {len(train_sents)}")
print(f"Val sentences: {len(val_sents)}")
print(f"Test sentences: {len(test_sents)}")

# ==================== 3. Word tokenization with <BOS>/<EOS> ====================
def tokenize_with_special_tokens(sentences):
    tokenized = []
    for sent in sentences:
        # Use NLTK word_tokenize for proper English tokenization (handles punctuation, contractions, etc.)
        words = nltk.word_tokenize(sent)
        # Optional: lowercase everything for better generalization (common in English LMs)
        words = [w.lower() for w in words]
        tokenized.append(['<BOS>'] + words + ['<EOS>'])
    return tokenized

print("Word tokenizing with NLTK...")
train_tokens = tokenize_with_special_tokens(train_sents)
val_tokens = tokenize_with_special_tokens(val_sents)
test_tokens = tokenize_with_special_tokens(test_sents)

# ==================== 4. Build vocabulary and replace <UNK> ====================
all_train_words = [word for sent in train_tokens for word in sent]
word_counts = Counter(all_train_words)

VOCAB_SIZE = 10000  # You can increase this for English (e.g., 20000) since it's richer
top_words = {word for word, _ in word_counts.most_common(VOCAB_SIZE)}
vocab = top_words | {'<BOS>', '<EOS>', '<UNK>'}

print(f"Vocabulary size: {len(vocab)} (top {VOCAB_SIZE} + special tokens)")

def replace_unk(tokenized_sents):
    return [[word if word in vocab else '<UNK>' for word in sent] for sent in tokenized_sents]

train_tokens = replace_unk(train_tokens)
val_tokens = replace_unk(val_tokens)
test_tokens = replace_unk(test_tokens)

# ==================== 5. Build n-gram counts ====================
def build_ngram_counts(tokenized_sents, n):
    counts = defaultdict(int)
    for sent in tokenized_sents:
        for i in range(len(sent) - n + 1):
            ngram = tuple(sent[i:i+n])
            counts[ngram] += 1
    return counts

print("Building n-gram counts...")
unigram_counts = build_ngram_counts(train_tokens, 1)
bigram_counts = build_ngram_counts(train_tokens, 2)
trigram_counts = build_ngram_counts(train_tokens, 3)
fourgram_counts = build_ngram_counts(train_tokens, 4)

total_tokens = sum(unigram_counts.values())

# ==================== Precompute context counts ====================
print("Precomputing context counts for smoothing...")
context4_counts = defaultdict(int)
for ngram, cnt in fourgram_counts.items():
    context4_counts[ngram[:-1]] += cnt

context3_counts = defaultdict(int)
for ngram, cnt in trigram_counts.items():
    context3_counts[ngram[:-1]] += cnt

context2_counts = defaultdict(int)
for ngram, cnt in bigram_counts.items():
    context2_counts[ngram[:-1]] += cnt

print("Setup complete! Ready for models.\n")

# ==================== 6. Perplexity Calculation ====================
def perplexity(tokenized_sents, prob_func, *args):
    log_prob = 0.0
    total_words = 0
    for sent in tokenized_sents:
        history = []
        for word in sent:
            if len(history) >= 3:  # We only compute prob after having 3 history words
                p = prob_func(history[-3:], word, *args)
                if p <= 0:
                    p = 1e-12
                log_prob += math.log(p)
                total_words += 1
            history.append(word)
    return math.exp(-log_prob / total_words) if total_words > 0 else float('inf')

# ==================== 7. LM1: Backoff (unsmoothed) ====================
def backoff_prob(history_3, word):
    # 4-gram
    context4 = tuple(history_3)
    count4 = fourgram_counts.get(context4 + (word,), 0)
    context4_count = context4_counts.get(context4, 0)
    if context4_count > 0 and count4 > 0:
        return count4 / context4_count
    
    # 3-gram
    context3 = tuple(history_3[-2:])
    count3 = trigram_counts.get(context3 + (word,), 0)
    context3_count = context3_counts.get(context3, 0)
    if context3_count > 0 and count3 > 0:
        return count3 / context3_count
    
    # 2-gram
    context2 = (history_3[-1],)
    count2 = bigram_counts.get(context2 + (word,), 0)
    context2_count = context2_counts.get(context2, 0)
    if context2_count > 0 and count2 > 0:
        return count2 / context2_count
    
    # Unigram fallback
    uni_count = unigram_counts.get((word,), 0)
    if uni_count > 0:
        return uni_count / total_tokens
    
    return 1.0 / len(vocab)  # Very small probability for OOV

print("LM1 (Backoff) ready")

# ==================== 8. LM2: Interpolation with Add-k Smoothing ====================
def addk_smoothed_prob(word, context, n, k):
    ngram_dict = {1: unigram_counts, 2: bigram_counts, 3: trigram_counts, 4: fourgram_counts}[n]
    
    if n == 1:
        context_count = total_tokens
    elif n == 2:
        context_count = context2_counts.get(context, 0)
    elif n == 3:
        context_count = context3_counts.get(context, 0)
    elif n == 4:
        context_count = context4_counts.get(context, 0)
    else:
        context_count = 0
    
    count = ngram_dict.get(context + (word,), 0)
    return (count + k) / (context_count + k * len(vocab))

def interpolation_prob(history_3, word, lambdas, k):
    l4, l3, l2, l1 = lambdas
    p4 = addk_smoothed_prob(word, tuple(history_3), 4, k)
    p3 = addk_smoothed_prob(word, tuple(history_3[-2:]), 3, k)
    p2 = addk_smoothed_prob(word, (history_3[-1],), 2, k)
    p1 = addk_smoothed_prob(word, (), 1, k)
    return l4 * p4 + l3 * p3 + l2 * p2 + l1 * p1

print("LM2 (Interpolation + Add-k) ready\n")

# ==================== 9. Tune LM2 on Validation ====================
print("Tuning LM2 hyperparameters on validation set...")
best_perp = float('inf')
best_lambdas = None
best_k = None

ks = [0.001, 0.01, 0.05, 0.1]
lambda_sets = [
    (0.6, 0.25, 0.1, 0.05),
    (0.5, 0.3, 0.15, 0.05),
    (0.4, 0.35, 0.2, 0.05),
    (0.7, 0.2, 0.08, 0.02),
    (0.3, 0.4, 0.2, 0.1)
]

for k in ks:
    for lam in lambda_sets:
        perp = perplexity(val_tokens, interpolation_prob, lam, k)
        print(f"  Trying λ={lam}, k={k} → Val Perplexity: {perp:.2f}")
        if perp < best_perp:
            best_perp = perp
            best_lambdas = lam
            best_k = k

print(f"\nBest validation perplexity: {best_perp:.2f} with λ={best_lambdas}, k={best_k}\n")

# ==================== 10. Evaluate on Test Set ====================
print("Evaluating on test set...")
lm1_perp = perplexity(test_tokens, backoff_prob)
lm2_perp = perplexity(test_tokens, interpolation_prob, best_lambdas, best_k)

print(f"LM1 (Backoff) Test Perplexity: {lm1_perp:.2f}")
print(f"LM2 (Interpolation + Add-k) Test Perplexity: {lm2_perp:.2f}\n")

# ==================== 11. Text Generation ====================
def generate_text(prob_func, max_words=50, *args):
    sent = ['<BOS>']
    while len(sent) < max_words + 1 and sent[-1] != '<EOS>':
        history_3 = sent[-3:] if len(sent) >= 3 else sent
        probs = {}
        for w in vocab:
            if w in {'<BOS>', '<UNK>', '<EOS>'}:
                continue
            p = prob_func(history_3, w, *args)
            if p > 1e-8:  # Filter very low probs for speed
                probs[w] = p
        if not probs:
            break
        words = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights /= weights.sum()
        next_word = np.random.choice(words, p=weights)
        sent.append(next_word)
    
    # Clean output: remove special tokens and join
    cleaned = [w for w in sent if w not in {'<BOS>', '<EOS>'}]
    return ' '.join(cleaned).strip()

print("Generating English text samples...")
np.random.seed(42)

print("\nLM1 (Backoff) generated text:")
print(generate_text(backoff_prob, 50))

print("\nLM2 (Best Interpolation) generated text - Sample 1:")
print(generate_text(interpolation_prob, 50, best_lambdas, best_k))

print("\nLM2 generated text - Sample 2:")
print(generate_text(interpolation_prob, 50, best_lambdas, best_k))

print("\nAll done! English N-gram Language Model project complete!")
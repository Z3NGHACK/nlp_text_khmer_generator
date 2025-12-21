# main_project.py - OPTIMIZED VERSION (NO VOCAB FILTERING)
import random
from collections import Counter, defaultdict
import math
import numpy as np
from khmernltk import sentence_tokenize, word_tokenize
from datetime import datetime

# ==================== CONFIGURATION ====================
CONFIG = {
    'vocab_size': 10000,
    'seed': 42,
    'max_gen_words': 50,
    'gen_temperature': 0.8,
    'n_gen_samples': 3,
}

# ==================== OUTPUT LOGGING ====================
class ResultLogger:
    def __init__(self, filename='results.txt'):
        self.filename = filename
        self.logs = []
        
    def log(self, message):
        print(message)
        self.logs.append(message)
    
    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.logs))
        print(f"\n✓ Results saved to {self.filename}")

logger = ResultLogger()
logger.log("=" * 80)
logger.log(f"OPTIMIZED N-GRAM LANGUAGE MODEL - Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.log("=" * 80)

# ==================== 1. Load Data ====================
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

logger.log("\n[1] Loading data...")
train_text = load_text('train.txt')
val_text = load_text('val.txt')
test_text = load_text('test.txt')

# ==================== 2. Sentence Tokenization ====================
logger.log("[2] Sentence tokenizing...")
train_sents = sentence_tokenize(train_text)
val_sents = sentence_tokenize(val_text)
test_sents = sentence_tokenize(test_text)

logger.log(f"    Train: {len(train_sents)} sentences")
logger.log(f"    Val:   {len(val_sents)} sentences")
logger.log(f"    Test:  {len(test_sents)} sentences")

# ==================== 3. Word Tokenization ====================
def tokenize_with_special_tokens(sentences):
    tokenized = []
    for sent in sentences:
        words = word_tokenize(sent)
        if words:
            tokenized.append(['<BOS>'] + words + ['<EOS>'])
    return tokenized

logger.log("[3] Word tokenizing...")
train_tokens = tokenize_with_special_tokens(train_sents)
val_tokens = tokenize_with_special_tokens(val_sents)
test_tokens = tokenize_with_special_tokens(test_sents)

# ==================== 4. Build Vocabulary (NO FILTERING!) ====================
logger.log("[4] Building vocabulary...")
all_train_words = [word for sent in train_tokens for word in sent]
word_counts = Counter(all_train_words)

# Keep ALL words - no filtering
top_words = {word for word, _ in word_counts.most_common(CONFIG['vocab_size'])}
vocab = top_words | {'<BOS>', '<EOS>', '<UNK>'}

logger.log(f"    Total unique words: {len(word_counts)}")
logger.log(f"    Vocabulary size: {len(vocab)}")

def replace_unk(tokenized_sents):
    return [[word if word in vocab else '<UNK>' for word in sent] for sent in tokenized_sents]

train_tokens = replace_unk(train_tokens)
val_tokens = replace_unk(val_tokens)
test_tokens = replace_unk(test_tokens)

# Calculate UNK rates
def calc_unk_rate(tokens):
    total = sum(len(sent) for sent in tokens)
    unks = sum(1 for sent in tokens for word in sent if word == '<UNK>')
    return (unks / total * 100) if total > 0 else 0

logger.log(f"    UNK rate - Val: {calc_unk_rate(val_tokens):.2f}%, Test: {calc_unk_rate(test_tokens):.2f}%")

# ==================== 5. Build N-gram Counts (Up to 5-grams) ====================
def build_ngram_counts(tokenized_sents, n):
    counts = defaultdict(int)
    for sent in tokenized_sents:
        for i in range(len(sent) - n + 1):
            ngram = tuple(sent[i:i+n])
            counts[ngram] += 1
    return counts

logger.log("[5] Building n-gram counts...")
unigram_counts = build_ngram_counts(train_tokens, 1)
bigram_counts = build_ngram_counts(train_tokens, 2)
trigram_counts = build_ngram_counts(train_tokens, 3)
fourgram_counts = build_ngram_counts(train_tokens, 4)
fivegram_counts = build_ngram_counts(train_tokens, 5)

total_tokens = sum(unigram_counts.values())

logger.log(f"    Unigrams:  {len(unigram_counts):,}")
logger.log(f"    Bigrams:   {len(bigram_counts):,}")
logger.log(f"    Trigrams:  {len(trigram_counts):,}")
logger.log(f"    4-grams:   {len(fourgram_counts):,}")
logger.log(f"    5-grams:   {len(fivegram_counts):,}")

# ==================== 6. Precompute Context Counts ====================
logger.log("[6] Precomputing context counts...")

context5_counts = defaultdict(int)
for ngram, cnt in fivegram_counts.items():
    context5_counts[ngram[:-1]] += cnt

context4_counts = defaultdict(int)
for ngram, cnt in fourgram_counts.items():
    context4_counts[ngram[:-1]] += cnt

context3_counts = defaultdict(int)
for ngram, cnt in trigram_counts.items():
    context3_counts[ngram[:-1]] += cnt

context2_counts = defaultdict(int)
for ngram, cnt in bigram_counts.items():
    context2_counts[ngram[:-1]] += cnt

# ==================== 7. Perplexity Calculation ====================
def perplexity(tokenized_sents, prob_func, *args):
    log_prob = 0.0
    total_words = 0
    
    for sent in tokenized_sents:
        for i in range(1, len(sent)):
            word = sent[i]
            history = sent[max(0, i-4):i]
            
            p = prob_func(history, word, *args)
            if p <= 0:
                p = 1e-12
            log_prob += math.log(p)
            total_words += 1
    
    return math.exp(-log_prob / total_words) if total_words > 0 else float('inf')

# ==================== 8. LM1: Stupid Backoff (5-gram) ====================
def stupid_backoff_prob(history, word, alpha=0.4):
    """Stupid Backoff with 5-gram support"""
    # 5-gram
    if len(history) >= 4:
        context5 = tuple(history[-4:])
        count5 = fivegram_counts.get(context5 + (word,), 0)
        context5_count = context5_counts.get(context5, 0)
        if context5_count > 0 and count5 > 0:
            return count5 / context5_count
    
    # 4-gram
    if len(history) >= 3:
        context4 = tuple(history[-3:])
        count4 = fourgram_counts.get(context4 + (word,), 0)
        context4_count = context4_counts.get(context4, 0)
        if context4_count > 0 and count4 > 0:
            return alpha * (count4 / context4_count)
    
    # 3-gram
    if len(history) >= 2:
        context3 = tuple(history[-2:])
        count3 = trigram_counts.get(context3 + (word,), 0)
        context3_count = context3_counts.get(context3, 0)
        if context3_count > 0 and count3 > 0:
            return (alpha ** 2) * (count3 / context3_count)
    
    # 2-gram
    if len(history) >= 1:
        context2 = (history[-1],)
        count2 = bigram_counts.get(context2 + (word,), 0)
        context2_count = context2_counts.get(context2, 0)
        if context2_count > 0 and count2 > 0:
            return (alpha ** 3) * (count2 / context2_count)
    
    # Unigram
    uni_count = unigram_counts.get((word,), 0)
    if uni_count > 0:
        return (alpha ** 4) * (uni_count / total_tokens)
    
    return (alpha ** 5) / len(vocab)

logger.log("[7] LM1 (Stupid Backoff - 5-gram) initialized")

# ==================== 9. LM2: Interpolation with Optimized Add-k ====================
def interpolation_5gram_prob(history, word, lambdas, k):
    """5-gram interpolation with add-k smoothing"""
    l5, l4, l3, l2, l1 = lambdas
    
    def smooth_prob(word, context, n, k):
        ngram_dict = {1: unigram_counts, 2: bigram_counts, 3: trigram_counts, 
                      4: fourgram_counts, 5: fivegram_counts}[n]
        context_dict = {1: total_tokens, 2: context2_counts, 3: context3_counts,
                        4: context4_counts, 5: context5_counts}
        
        if n == 1:
            context_count = total_tokens
        else:
            context_count = context_dict[n].get(context, 0)
        
        count = ngram_dict.get(context + (word,), 0)
        
        if context_count == 0:
            return k / (k * len(vocab))
        
        return (count + k) / (context_count + k * len(vocab))
    
    # Calculate probabilities
    p5 = smooth_prob(word, tuple(history[-4:]), 5, k) if len(history) >= 4 else 0
    p4 = smooth_prob(word, tuple(history[-3:]), 4, k) if len(history) >= 3 else 0
    p3 = smooth_prob(word, tuple(history[-2:]), 3, k) if len(history) >= 2 else 0
    p2 = smooth_prob(word, (history[-1],), 2, k) if len(history) >= 1 else 0
    p1 = smooth_prob(word, (), 1, k)
    
    # Adjust lambdas based on available history
    if len(history) < 4:
        l5 = 0
    if len(history) < 3:
        l4 = 0
    if len(history) < 2:
        l3 = 0
    if len(history) < 1:
        l2 = 0
    
    # Renormalize
    total_lambda = l5 + l4 + l3 + l2 + l1
    if total_lambda > 0:
        l5, l4, l3, l2, l1 = l5/total_lambda, l4/total_lambda, l3/total_lambda, l2/total_lambda, l1/total_lambda
    
    result = l5 * p5 + l4 * p4 + l3 * p3 + l2 * p2 + l1 * p1
    return max(result, 1e-12)

logger.log("[8] LM2 (5-gram Interpolation + Add-k) initialized")

# ==================== 10. LM3: Modified Kneser-Ney (4-gram) ====================
def build_continuation_counts():
    """Build continuation counts for Kneser-Ney"""
    continuation_counts = defaultdict(int)
    for (w1, w2) in bigram_counts.keys():
        continuation_counts[(w2,)] += 1
    return continuation_counts

continuation_counts = build_continuation_counts()

def modified_kneser_ney_prob(history, word, discount=0.75):
    """Modified Kneser-Ney with better fallback"""
    # 4-gram
    if len(history) >= 3:
        context4 = tuple(history[-3:])
        count4 = fourgram_counts.get(context4 + (word,), 0)
        context4_count = context4_counts.get(context4, 0)
        
        if context4_count > 0:
            num_continuations = len([w for w in vocab if fourgram_counts.get(context4 + (w,), 0) > 0])
            lambda4 = (discount * num_continuations) / context4_count
            prob4 = max(count4 - discount, 0) / context4_count
        else:
            prob4 = 0
            lambda4 = 1.0
    else:
        prob4 = 0
        lambda4 = 1.0
    
    # 3-gram
    if len(history) >= 2:
        context3 = tuple(history[-2:])
        count3 = trigram_counts.get(context3 + (word,), 0)
        context3_count = context3_counts.get(context3, 0)
        
        if context3_count > 0:
            num_continuations = len([w for w in vocab if trigram_counts.get(context3 + (w,), 0) > 0])
            lambda3 = (discount * num_continuations) / context3_count
            prob3 = max(count3 - discount, 0) / context3_count
        else:
            prob3 = 0
            lambda3 = 1.0
    else:
        prob3 = 0
        lambda3 = 1.0
    
    # 2-gram
    if len(history) >= 1:
        context2 = (history[-1],)
        count2 = bigram_counts.get(context2 + (word,), 0)
        context2_count = context2_counts.get(context2, 0)
        
        if context2_count > 0:
            prob2 = max(count2 - discount, 0) / context2_count
            num_continuations = len([w for w in vocab if bigram_counts.get(context2 + (w,), 0) > 0])
            lambda2 = (discount * num_continuations) / context2_count
        else:
            prob2 = 0
            lambda2 = 1.0
    else:
        prob2 = 0
        lambda2 = 1.0
    
    # Unigram with continuation probability
    pcont = continuation_counts.get((word,), 0) / len(bigram_counts) if len(bigram_counts) > 0 else 1.0 / len(vocab)
    
    # Combine with interpolation
    result = prob4 + lambda4 * (prob3 + lambda3 * (prob2 + lambda2 * pcont))
    return max(result, 1e-12)

logger.log("[9] LM3 (Modified Kneser-Ney - 4-gram) initialized")

# ==================== 11. Hyperparameter Tuning ====================
logger.log("\n" + "=" * 80)
logger.log("[10] HYPERPARAMETER TUNING")
logger.log("=" * 80)

# Tune LM1
logger.log("\nTuning LM1 (Stupid Backoff)...")
best_lm1_alpha = 0.4
best_lm1_perp = float('inf')

for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
    perp = perplexity(val_tokens, stupid_backoff_prob, alpha)
    logger.log(f"  alpha={alpha:.1f} → Val Perplexity: {perp:.2f}")
    if perp < best_lm1_perp:
        best_lm1_perp = perp
        best_lm1_alpha = alpha

logger.log(f"✓ Best: alpha={best_lm1_alpha}, Perplexity={best_lm1_perp:.2f}")

# Tune LM2
logger.log("\nTuning LM2 (5-gram Interpolation)...")
best_lm2_perp = float('inf')
best_lm2_lambdas = None
best_lm2_k = None

ks = [0.00005, 0.0001, 0.0005]
lambda_sets = [
    (0.5, 0.25, 0.15, 0.08, 0.02),
    (0.4, 0.3, 0.2, 0.08, 0.02),
    (0.35, 0.3, 0.2, 0.1, 0.05),
    (0.3, 0.3, 0.2, 0.15, 0.05),
    (0.25, 0.25, 0.25, 0.15, 0.10),
]

for k in ks:
    for lam in lambda_sets:
        perp = perplexity(val_tokens, interpolation_5gram_prob, lam, k)
        logger.log(f"  λ={lam}, k={k:.5f} → Perplexity: {perp:.2f}")
        if perp < best_lm2_perp:
            best_lm2_perp = perp
            best_lm2_lambdas = lam
            best_lm2_k = k

logger.log(f"✓ Best: λ={best_lm2_lambdas}, k={best_lm2_k}, Perplexity={best_lm2_perp:.2f}")

# Tune LM3
logger.log("\nTuning LM3 (Modified Kneser-Ney)...")
best_lm3_discount = 0.75
best_lm3_perp = float('inf')

for discount in [0.5, 0.6, 0.75, 0.85, 0.9]:
    perp = perplexity(val_tokens, modified_kneser_ney_prob, discount)
    logger.log(f"  discount={discount:.2f} → Val Perplexity: {perp:.2f}")
    if perp < best_lm3_perp:
        best_lm3_perp = perp
        best_lm3_discount = discount

logger.log(f"✓ Best: discount={best_lm3_discount}, Perplexity={best_lm3_perp:.2f}")

# ==================== 12. Final Evaluation ====================
logger.log("\n" + "=" * 80)
logger.log("[11] FINAL TEST SET EVALUATION")
logger.log("=" * 80)

lm1_test = perplexity(test_tokens, stupid_backoff_prob, best_lm1_alpha)
lm2_test = perplexity(test_tokens, interpolation_5gram_prob, best_lm2_lambdas, best_lm2_k)
lm3_test = perplexity(test_tokens, modified_kneser_ney_prob, best_lm3_discount)

logger.log(f"\nLM1 (Stupid Backoff):      {lm1_test:.2f}")
logger.log(f"LM2 (5-gram Interpolation): {lm2_test:.2f}")
logger.log(f"LM3 (Modified Kneser-Ney):  {lm3_test:.2f}")

models = [("LM1", lm1_test), ("LM2", lm2_test), ("LM3", lm3_test)]
best_model = min(models, key=lambda x: x[1])
worst_model = max(models, key=lambda x: x[1])

improvement = ((worst_model[1] - best_model[1]) / worst_model[1]) * 100
logger.log(f"\n✓ Winner: {best_model[0]} with perplexity {best_model[1]:.2f}")
logger.log(f"  Improvement over worst: {improvement:.1f}%")

# ==================== 13. Text Generation ====================
def generate_text(prob_func, max_words=50, temperature=0.8, min_words=15, *args):
    """Generate text with better control"""
    sent = ['<BOS>']
    attempts = 0
    
    while attempts < max_words * 2:
        attempts += 1
        history = sent[-4:] if len(sent) >= 4 else sent
        
        probs = {}
        for w in vocab:
            if w in {'<BOS>', '<UNK>'}:
                continue
            if w == '<EOS>' and len(sent) < min_words:
                continue
            
            p = prob_func(history, w, *args)
            if p > 0:
                probs[w] = p ** (1.0 / temperature)
        
        if not probs:
            break
        
        words = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights /= weights.sum()
        next_word = np.random.choice(words, p=weights)
        sent.append(next_word)
        
        if next_word == '<EOS>' and len(sent) >= min_words:
            break
        if len(sent) >= max_words:
            break
    
    return ' '.join([w for w in sent if w not in {'<BOS>', '<EOS>'}])

logger.log("\n" + "=" * 80)
logger.log("[12] TEXT GENERATION SAMPLES")
logger.log("=" * 80)

np.random.seed(CONFIG['seed'])

logger.log("\n--- LM1 (Stupid Backoff) ---")
for i in range(CONFIG['n_gen_samples']):
    text = generate_text(stupid_backoff_prob, CONFIG['max_gen_words'], 
                        CONFIG['gen_temperature'], 15, best_lm1_alpha)
    logger.log(f"\n[{i+1}] {text}")

logger.log("\n\n--- LM2 (5-gram Interpolation) ---")
for i in range(CONFIG['n_gen_samples']):
    text = generate_text(interpolation_5gram_prob, CONFIG['max_gen_words'], 
                        CONFIG['gen_temperature'], 15, best_lm2_lambdas, best_lm2_k)
    logger.log(f"\n[{i+1}] {text}")

logger.log("\n\n--- LM3 (Modified Kneser-Ney) ---")
for i in range(CONFIG['n_gen_samples']):
    text = generate_text(modified_kneser_ney_prob, CONFIG['max_gen_words'], 
                        CONFIG['gen_temperature'], 15, best_lm3_discount)
    logger.log(f"\n[{i+1}] {text}")

# ==================== 14. Summary ====================
logger.log("\n" + "=" * 80)
logger.log("[13] SUMMARY")
logger.log("=" * 80)

logger.log(f"\nDataset: {len(train_sents)}/{len(val_sents)}/{len(test_sents)} sentences")
logger.log(f"Vocabulary: {len(vocab):,} words")
logger.log(f"Total training tokens: {total_tokens:,}")

logger.log(f"\nTest Perplexities:")
logger.log(f"  LM1 (Stupid Backoff):      {lm1_test:.2f}")
logger.log(f"  LM2 (5-gram Interpolation): {lm2_test:.2f}")
logger.log(f"  LM3 (Modified Kneser-Ney):  {lm3_test:.2f}")

logger.log(f"\nBest Model: {best_model[0]}")
logger.log(f"Best Parameters:")
logger.log(f"  LM1: alpha={best_lm1_alpha}")
logger.log(f"  LM2: λ={best_lm2_lambdas}, k={best_lm2_k}")
logger.log(f"  LM3: discount={best_lm3_discount}")

logger.log("\n" + "=" * 80)
logger.log("EXPERIMENT COMPLETED!")
logger.log("=" * 80)

logger.save()
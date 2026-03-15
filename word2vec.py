"""
Word2Vec (Skip-Gram & CBOW) with Negative Sampling

Usage:
  python word2vec.py text8
"""

import numpy as np


# Numerical functions

def sigmoid(x):
    """Sigmoid with clipping to avoid overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def log_sigmoid(x):
    """ log(sigmoid(x)) = -log(1 + exp(-x))."""
    return -np.logaddexp(0.0, -x)

# Functions for data processing

def tokenize(text):
    """Lowercase text and split into words."""
    return text.lower().split()


def build_vocab(tokens, min_count=5):
    """Build vocabulary filtered by min_count.

    Returns:
        word2idx  – dictionary mapping word -> integer index
        idx2word  – dictionary mapping index -> word
        freqs     – np.array of raw counts, indexed by word index
    """

    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1 # Count occurences of each word

    vocab = [(w, c) for w, c in counts.items() if c >= min_count] # Keep only words appearing at least min_count times
    vocab.sort(key=lambda wc: -wc[1])  # most frequent first

    word2idx = {w: i for i, (w, _) in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    freqs = np.array([c for _, c in vocab], dtype=np.float64)

    return word2idx, idx2word, freqs


def subsample_keep_prob(freqs, t=1e-5):
    """Probability of keeping each word (subsampling frequent words)

    P(keep_i) = sqrt(t / f_i), where f_i = count_i / total_count
    Words rarer than threshold t are always kept
    """
    f = freqs / freqs.sum()
    prob = np.sqrt(t / f)
    return np.minimum(prob, 1.0)


def build_neg_table(freqs, table_size=10_000_000):
    """Precompute a table for negative sampling

    Each word's representation in the table is proportional to freq^(3/4)
    """
    weights = freqs ** 0.75
    weights /= weights.sum()

    # Each word gets a number of slots proportional to its weight
    counts = np.floor(weights * table_size).astype(np.int64) # Converts each word's probability into a number of table slots
    counts[-1] = table_size - counts[:-1].sum()  # Fix rounding. Ensure total slots equal table_size
    return np.repeat(np.arange(len(freqs), dtype=np.int64), counts) # Fill the table with the word indices


# Training functions

def train_skipgram(corpus_path, embed_dim=100, window_size=5, num_neg=5,
                   lr0=0.025, min_lr=1e-4, min_count=5, epochs=5, t=1e-5):
    """End-to-end training of skip-gram with negative sampling"""

    # Preprocessing with previously defined functions
    with open(corpus_path, "r") as f:
        text = f.read()
    tokens = tokenize(text)
    print(f"  Total tokens: {len(tokens):,}")

    # build vocabulary, word -> index and index -> word dictionaries, words frequency array
    word2idx, idx2word, freqs = build_vocab(tokens, min_count)
    V = len(word2idx)
    print(f"  Vocab size: {V:,}")

    # build corpus (filter min_count words), subsampling array (probability of keeping each word), negative sampling table
    corpus = np.array([word2idx[w] for w in tokens if w in word2idx], dtype=np.int64)
    keep_prob = subsample_keep_prob(freqs, t)
    neg_table = build_neg_table(freqs)

    # Initialize embeddings
    W_center = np.random.uniform(-0.01, 0.01, (V, embed_dim)) # Matrix of center words embeddings. Small numbers near zero.
    W_context = np.zeros((V, embed_dim)) # Matrix of context words embeddings. Initialized to zero.

    # Training
    total_steps = epochs * len(corpus) * window_size # needed for learning rate decay
    global_step = 0

    for epoch in range(epochs):
        # Subsample frequent words. Called at the start of each epoch, so each new run would get a new subsampled corpus
        mask = np.random.random(len(corpus)) < keep_prob[corpus]
        sub_corpus = corpus[mask]

        epoch_loss = 0.0
        epoch_pairs = 0

        for i in range(len(sub_corpus)):
            center = sub_corpus[i] # pick this word as the center word

            # Define the context window size
            start = max(0, i - window_size)
            end = min(len(sub_corpus), i + window_size + 1)

            for j in range(start, end): # loop over all words in the window
                if j == i:
                    continue
                context = sub_corpus[j] # pick this word as the context word

                # Learning rate decay
                lr = max(lr0 * (1.0 - global_step / total_steps), min_lr)
                global_step += 1

                # Sample negatives
                neg_idx = neg_table[np.random.randint(0, len(neg_table), size=num_neg)]

                # Forward pass
                v_c = W_center[center] # embedding of the center word
                u_o = W_context[context] # embedding of the context word
                u_neg = W_context[neg_idx] # embeddings of the negative samples

                # Dot products for scoring
                pos_dot = u_o @ v_c
                neg_dots = u_neg @ v_c

                # Convert scores to probabilities using sigmoid function
                pos_sig = sigmoid(pos_dot)
                neg_sigs = sigmoid(neg_dots)

                # Calculate training loss at this step
                loss = -log_sigmoid(pos_dot) - np.sum(log_sigmoid(-neg_dots))
                epoch_loss += loss
                epoch_pairs += 1

                # Gradients of the embedding vectors
                grad_vc = (pos_sig - 1.0) * u_o + neg_sigs @ u_neg
                grad_uo = (pos_sig - 1.0) * v_c
                grad_un = np.outer(neg_sigs, v_c)

                # SGD updates
                W_center[center] -= lr * grad_vc
                W_context[context] -= lr * grad_uo
                W_context[neg_idx] -= lr * grad_un

            # Some prints for monitoring and output
            if (i + 1) % 100_000 == 0:
                avg_loss = epoch_loss / max(epoch_pairs, 1)
                print(f"  Epoch {epoch+1} ; {i+1:,}/{len(sub_corpus):,} words ; loss {avg_loss:.4f} ; lr {lr:.6f}")

        avg_loss = epoch_loss / max(epoch_pairs, 1)
        print(f"  Epoch {epoch+1}/{epochs} done. Average loss: {avg_loss:.4f})")

    return W_center, word2idx, idx2word


def train_cbow(corpus_path, embed_dim=100, window_size=5, num_neg=5,
               lr0=0.025, min_lr=1e-4, min_count=5, epochs=5, t=1e-5):
    """End-to-end training of CBOW with negative sampling"""

    # Preprocessing with previously defined functions
    with open(corpus_path, "r") as f:
        text = f.read()
    tokens = tokenize(text)
    print(f"  Total tokens: {len(tokens):,}")

    # build vocabulary, word -> index and index -> word dictionaries, words frequency array
    word2idx, idx2word, freqs = build_vocab(tokens, min_count)
    V = len(word2idx)
    print(f"  Vocab size:   {V:,}")

    # build corpus (filter min_count words), subsampling array (probability of keeping each word), negative sampling table
    corpus = np.array([word2idx[w] for w in tokens if w in word2idx], dtype=np.int64)
    keep_prob = subsample_keep_prob(freqs, t)
    neg_table = build_neg_table(freqs)

    # Initialize embeddings
    W_center = np.random.uniform(-0.01, 0.01, (V, embed_dim)) # Matrix of center words embeddings. Small numbers near zero.
    W_context = np.zeros((V, embed_dim)) # Matrix of context words embeddings. Initialized to zero.

    # Training
    total_steps = epochs * len(corpus) * window_size # needed for learning rate decay
    global_step = 0

    for epoch in range(epochs):
        # Subsample frequent words. Called at the start of each epoch, so each new run would get a new subsampled corpus
        mask = np.random.random(len(corpus)) < keep_prob[corpus]
        sub_corpus = corpus[mask]

        epoch_loss = 0.0
        epoch_pairs = 0

        for i in range(len(sub_corpus)):
            center = sub_corpus[i] # pick this word as the center word

            # Define the context window size
            start = max(0, i - window_size)
            end = min(len(sub_corpus), i + window_size + 1)

            # Collect context indices (excluding center position)
            ctx_idx = [sub_corpus[j] for j in range(start, end) if j != i]
            if len(ctx_idx) == 0:
                continue
            ctx_idx = np.array(ctx_idx, dtype=np.int64)

            # Learning rate decay
            lr = max(lr0 * (1.0 - global_step / total_steps), min_lr)
            global_step += 1

            # Sample negatives
            neg_idx = neg_table[np.random.randint(0, len(neg_table), size=num_neg)]

            # Forward pass
            v_hat = W_center[ctx_idx].mean(axis=0) # average embeddings of the context window
            u_c = W_context[center] # embedding of the center word
            u_neg = W_context[neg_idx] # embeddings of the negative samples

            # Dot products for scoring
            pos_dot = u_c @ v_hat
            neg_dots = u_neg @ v_hat

            # Convert scores to probabilities using sigmoid function
            pos_sig = sigmoid(pos_dot)
            neg_sigs = sigmoid(neg_dots)

            # Calculate training loss at this step
            loss = -log_sigmoid(pos_dot) - np.sum(log_sigmoid(-neg_dots))
            epoch_loss += loss
            epoch_pairs += 1

            # Gradients of the embedding vectors
            grad_vhat = (pos_sig - 1.0) * u_c + neg_sigs @ u_neg
            grad_uc = (pos_sig - 1.0) * v_hat
            grad_un = np.outer(neg_sigs, v_hat)

            # SGD updates. Distribute the average embeddings gradient equally to each context word
            grad_per_ctx = (lr / len(ctx_idx)) * grad_vhat
            for idx in ctx_idx:
                W_center[idx] -= grad_per_ctx
            W_context[center] -= lr * grad_uc
            W_context[neg_idx] -= lr * grad_un

            # Some prints for monitoring and output
            if (i + 1) % 100_000 == 0:
                avg_loss = epoch_loss / max(epoch_pairs, 1)
                print(f"  Epoch {epoch + 1} ; {i + 1:,}/{len(sub_corpus):,} words ; loss {avg_loss:.4f} ; lr {lr:.6f}")

        avg_loss = epoch_loss / max(epoch_pairs, 1)
        print(f"  Epoch {epoch + 1}/{epochs} done. Average loss: {avg_loss:.4f})")

    return W_center, word2idx, idx2word # return the center word trained embeddings, word to index dictionary, index to word dictionary


# Some evaluation functions for post-training checks

def cosine_similarity(W, vec):
    """Cosine similarity between each row of W and vector vec."""
    vec_norm = np.linalg.norm(vec) # L2 norm of the vector

    # if the vector is zero, return zero similarity
    if vec_norm == 0:
        return np.zeros(W.shape[0])

    W_norm = np.linalg.norm(W, axis=1) # L2 norm of each row of W
    W_norm = np.maximum(W_norm, 1e-12) # avoid division by zero
    sims = (W @ vec) / (W_norm * vec_norm) # cosine similarity
    return np.nan_to_num(sims, nan=0.0) # replace NaN with zero


def most_similar(word, W, word2idx, idx2word, top_k=10):
    """Find the top-k most similar words by cosine similarity."""

    vec = W[word2idx[word]] # lookup vector for the word
    sims = cosine_similarity(W, vec) # cosine similarity between the word vector and all other vectors in the vocabulary
    top = np.argsort(-sims) # indices of the most similar words

    print(f"Most similar to '{word}':")
    shown = 0
    for idx in top: #   Walk through words in order of decreasing similarity
        if idx == word2idx[word]:
            continue
        print(f"{idx2word[idx]:20s} {sims[idx]:.4f}") # print the word and its similarity score
        shown += 1
        if shown == top_k:
            break


# Main

def evaluate(W, word2idx, idx2word):
    print()
    print(" Similarity checks ")
    for word in ["king", "computer", "france"]:
        if word in word2idx:
            most_similar(word, W, word2idx, idx2word)
            print()

# useful helper function to call the models and evaluate them
def run_model(name, train_fn, corpus_path):
    print()
    print(name)
    print()
    W, word2idx, idx2word = train_fn(
        corpus_path=corpus_path,
        embed_dim=100,
        window_size=5,
        num_neg=5,
        lr0=0.025,
        epochs=1 # change to 3 for better results of CBOW
    )
    evaluate(W, word2idx, idx2word)


if __name__ == "__main__":
    corpus_path = "text8"

    run_model("SKIP-GRAM", train_skipgram, corpus_path)
    print()
    run_model("CBOW", train_cbow, corpus_path)

import numpy as np
import pickle
import time
from collections import Counter
from tqdm import tqdm
from word2vec import Word2Vec, NegativeSampler


def load_corpus(filepath, min_freq=3):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = text.lower().split()
    print(f"Raw tokens: {len(tokens)}")

    # count and filter rare words
    freq = Counter(tokens)
    tokens = [t for t in tokens if freq[t] >= min_freq]
    freq = Counter(tokens)  # recount after filtering

    # build vocab
    word2idx = {}
    idx2word = {}
    for i, (word, _) in enumerate(freq.most_common()):
        word2idx[word] = i
        idx2word[i] = word

    vocab_size = len(word2idx)
    print(f"After filtering (min_freq={min_freq}): {len(tokens)} tokens, vocab size: {vocab_size}")

    # convert to indices
    token_ids = [word2idx[t] for t in tokens]

    # word freq by index (for negative sampling)
    idx_freqs = {word2idx[w]: c for w, c in freq.items()}

    return token_ids, word2idx, idx2word, idx_freqs, vocab_size


def generate_skipgram_pairs(token_ids, window):
    pairs = []
    for i in range(len(token_ids)):
        center = token_ids[i]
        left = max(0, i - window)
        right = min(len(token_ids), i + window + 1)
        for j in range(left, right):
            if j == i:
                continue
            pairs.append((center, token_ids[j]))
    return pairs


def generate_cbow_samples(token_ids, window):
    samples = []
    for i in range(window, len(token_ids) - window):
        center = token_ids[i]
        ctx = []
        for j in range(i - window, i + window + 1):
            if j != i:
                ctx.append(token_ids[j])
        samples.append((ctx, center))
    return samples


def train_model(mode, token_ids, vocab_size, idx_freqs,
                embed_dim=100, window=5, neg_samples=5,
                epochs=10, lr=0.025):

    print(f"\n--- Training {mode.upper()} | dim={embed_dim}, window={window}, neg={neg_samples}, epochs={epochs} ---")

    model = Word2Vec(vocab_size, embed_dim, lr=lr)
    sampler = NegativeSampler(idx_freqs, vocab_size)

    if mode == 'skipgram':
        pairs = generate_skipgram_pairs(token_ids, window)
        print(f"Training pairs: {len(pairs)}")
    else:
        samples = generate_cbow_samples(token_ids, window)
        print(f"Training samples: {len(samples)}")

    lr_start = lr
    n_items = len(pairs) if mode == 'skipgram' else len(samples)
    total_steps = epochs * n_items
    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0

        if mode == 'skipgram':
            np.random.shuffle(pairs)
            pbar = tqdm(pairs, desc=f"Epoch {epoch+1}/{epochs}", unit="pair", leave=True)
            for center, context in pbar:
                negs = sampler.sample(neg_samples, exclude={center, context})
                loss = model.train_skipgram(center, context, negs)
                total_loss += loss
                step += 1
                model.lr = max(lr_start * (1 - step / total_steps), 1e-5)

                if step % 10000 == 0:
                    pbar.set_postfix(loss=f"{total_loss/((step-1) % n_items + 1):.4f}", lr=f"{model.lr:.5f}")

            avg_loss = total_loss / len(pairs)

        else:  # cbow
            np.random.shuffle(samples)
            pbar = tqdm(samples, desc=f"Epoch {epoch+1}/{epochs}", unit="sample", leave=True)
            for ctx, center in pbar:
                exclude_set = set(ctx) | {center}
                negs = sampler.sample(neg_samples, exclude=exclude_set)
                loss = model.train_cbow(ctx, center, negs)
                total_loss += loss
                step += 1
                model.lr = max(lr_start * (1 - step / total_steps), 1e-5)

                if step % 10000 == 0:
                    pbar.set_postfix(loss=f"{total_loss/((step-1) % n_items + 1):.4f}", lr=f"{model.lr:.5f}")

            avg_loss = total_loss / len(samples)

        epoch_losses.append(avg_loss)
        print(f"  => Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    return model, epoch_losses


def save_model(model, word2idx, idx2word, epoch_losses, mode, config_tag="default"):
    data = {
        'W_in': model.get_embeddings(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'epoch_losses': epoch_losses,
    }
    fname = f"model_{mode}_{config_tag}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved: {fname}")
    return fname


if __name__ == '__main__':
    corpus_file = 'corpus.txt'
    token_ids, word2idx, idx2word, idx_freqs, vocab_size = load_corpus(corpus_file, min_freq=3)

    # train skip-gram
    sg_model, sg_losses = train_model(
        'skipgram', token_ids, vocab_size, idx_freqs,
        embed_dim=100, window=5, neg_samples=5, epochs=10, lr=0.025
    )
    save_model(sg_model, word2idx, idx2word, sg_losses, 'skipgram',"raw")

    # train CBOW
    cbow_model, cbow_losses = train_model(
        'cbow', token_ids, vocab_size, idx_freqs,
        embed_dim=100, window=5, neg_samples=5, epochs=10, lr=0.025
    )
    save_model(cbow_model, word2idx, idx2word, cbow_losses, 'cbow','raw')

    print("\nDone. Both models trained and saved.")
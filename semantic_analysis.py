import numpy as np
import pickle
import sys


def load_model(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['W_in'], data['word2idx'], data['idx2word']


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def nearest_neighbors(word, embeddings, word2idx, idx2word, top_k=5):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return []
    idx = word2idx[word]
    vec = embeddings[idx]

    sims = []
    for i in range(len(embeddings)):
        if i == idx:
            continue
        sims.append((i, cosine_sim(vec, embeddings[i])))

    sims.sort(key=lambda x: x[1], reverse=True)
    results = [(idx2word[i], score) for i, score in sims[:top_k]]
    return results


def analogy(a, b, c, embeddings, word2idx, idx2word, top_k=5):
    # a : b :: c : ?
    # result = b - a + c
    for w in [a, b, c]:
        if w not in word2idx:
            print(f"'{w}' not in vocabulary")
            return []

    vec = embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]]
    exclude = {word2idx[a], word2idx[b], word2idx[c]}

    sims = []
    for i in range(len(embeddings)):
        if i in exclude:
            continue
        sims.append((i, cosine_sim(vec, embeddings[i])))

    sims.sort(key=lambda x: x[1], reverse=True)
    return [(idx2word[i], score) for i, score in sims[:top_k]]


def run_neighbor_analysis(embeddings, word2idx, idx2word, model_name):
    query_words = ['research', 'student', 'phd', 'exam', 'exams']
    print(f"\n{'='*60}")
    print(f"  Nearest Neighbors — {model_name}")
    print(f"{'='*60}")
    for w in query_words:
        neighbors = nearest_neighbors(w, embeddings, word2idx, idx2word, top_k=5)
        if neighbors:
            print(f"\n  {w}:")
            for word, sim in neighbors:
                print(f"    {word:20s} {sim:.4f}")


def run_analogy_experiments(embeddings, word2idx, idx2word, model_name):
    # define your analogies here - easy to add/change
    analogies = [
        ('dean', 'office', 'wardens',    'UG : BTech :: PG : ?'),
        # ('student', 'hostel', 'faculty', 'student : hostel :: faculty : ?'),
        # ('professor', 'teaching', 'researcher', 'professor : teaching :: researcher : ?'),
    ]

    print(f"\n{'='*60}")
    print(f"  Analogy Experiments — {model_name}")
    print(f"{'='*60}")
    for a, b, c, desc in analogies:
        print(f"\n  {desc}")
        results = analogy(a, b, c, embeddings, word2idx, idx2word, top_k=5)
        if results:
            for word, sim in results:
                print(f"    {word:20s} {sim:.4f}")


if __name__ == '__main__':
    # you can pass model files as arguments, or just use defaults
    sg_file = sys.argv[1] if len(sys.argv) > 1 else 'model_skipgram_embed_dim_300.pkl'
    cbow_file = sys.argv[2] if len(sys.argv) > 2 else 'model_cbow_embed_dim_300.pkl'

    for fpath, name in [(sg_file, 'Skip-gram'), (cbow_file, 'CBOW')]:
        emb, w2i, i2w = load_model(fpath)
        run_neighbor_analysis(emb, w2i, i2w, name)
        run_analogy_experiments(emb, w2i, i2w, name)
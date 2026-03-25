import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_model(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['W_in'], data['word2idx'], data['idx2word']


# word groups to visualize — edit these as you like
WORD_GROUPS = {
    'Academic Programs': ['btech', 'mtech', 'msc', 'phd', 'diploma', 'degree'],
    'Departments': ['cse', 'electrical', 'mechanical', 'physics', 'chemistry', 'mathematics'],
    'Roles': ['student', 'faculty', 'professor', 'researcher', 'dean', 'instructor'],
    'Administrative': ['semester', 'exam', 'registration', 'grade', 'credit', 'course'],
}

COLORS = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']


def collect_words_and_vectors(word_groups, embeddings, word2idx):
    words = []
    vectors = []
    groups = []
    group_names = []

    for gi, (gname, wlist) in enumerate(word_groups.items()):
        for w in wlist:
            if w in word2idx:
                words.append(w)
                vectors.append(embeddings[word2idx[w]])
                groups.append(gi)
                group_names.append(gname)
            else:
                print(f"  skipping '{w}' (not in vocab)")

    return words, np.array(vectors), groups, group_names


def plot_2d(coords, words, groups, title, filename):
    fig, ax = plt.subplots(figsize=(12, 9))
    plotted_groups = set()
    group_keys = list(WORD_GROUPS.keys())

    for i, (x, y) in enumerate(coords):
        gi = groups[i]
        label = group_keys[gi] if gi not in plotted_groups else None
        ax.scatter(x, y, c=COLORS[gi], s=80, zorder=3, label=label)
        ax.annotate(
            words[i], (x, y),
            fontsize=9, ha='center', va='bottom',
            xytext=(0, 6), textcoords='offset points'
        )
        plotted_groups.add(gi)

    ax.legend(fontsize=10, loc='best')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


def visualize_model(filepath, model_name):
    emb, w2i, i2w = load_model(filepath)
    words, vectors, groups, _ = collect_words_and_vectors(WORD_GROUPS, emb, w2i)

    if len(vectors) < 3:
        print(f"Not enough words found in vocab for {model_name}, skipping.")
        return

    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(vectors)
    plot_2d(
        pca_coords, words, groups,
        f'PCA — {model_name}',
        f'viz_pca_{model_name.lower().replace("-", "_").replace(" ", "_")}.png'
    )

    # t-SNE
    perp = min(5, len(vectors) - 1)  # perplexity must be < n_samples
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    tsne_coords = tsne.fit_transform(vectors)
    plot_2d(
        tsne_coords, words, groups,
        f't-SNE — {model_name}',
        f'viz_tsne_{model_name.lower().replace("-", "_").replace(" ", "_")}.png'
    )


if __name__ == '__main__':
    sg_file = sys.argv[1] if len(sys.argv) > 1 else 'model_skipgram_default.pkl'
    cbow_file = sys.argv[2] if len(sys.argv) > 2 else 'model_cbow_default.pkl'

    print("Visualizing Skip-gram embeddings...")
    visualize_model(sg_file, 'Skip-gram')

    print("\nVisualizing CBOW embeddings...")
    visualize_model(cbow_file, 'CBOW')
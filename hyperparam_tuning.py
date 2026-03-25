import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from train_word2vec import load_corpus, train_model, save_model


def run_sweep(param_name, values, token_ids, vocab_size, idx_freqs, word2idx, idx2word):
    # defaults
    defaults = {'embed_dim': 100, 'window': 5, 'neg_samples': 5, 'epochs': 10, 'lr': 0.025}
    results = {'skipgram': {}, 'cbow': {}}

    for val in values:
        config = defaults.copy()
        config[param_name] = val
        tag = f"{param_name}_{val}"
        print(f"\n===== {param_name} = {val} =====")

        for mode in ['skipgram', 'cbow']:
            model, losses = train_model(
                mode, token_ids, vocab_size, idx_freqs,
                embed_dim=config['embed_dim'],
                window=config['window'],
                neg_samples=config['neg_samples'],
                epochs=config['epochs'],
                lr=config['lr']
            )
            results[mode][val] = losses[-1]  # final epoch loss
            save_model(model, word2idx, idx2word, losses, mode, config_tag=tag)

    return results


def plot_results(param_name, values, results, ylabel='Final Epoch Loss'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values, [results['skipgram'][v] for v in values], 'o-', label='Skip-gram', color='#2196F3')
    ax.plot(values, [results['cbow'][v] for v in values], 's--', label='CBOW', color='#E91E63')
    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Effect of {param_name.replace("_", " ").title()} on Training Loss', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"sweep_{param_name}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Plot saved: {fname}")


if __name__ == '__main__':
    corpus_file = 'corpus.txt'
    token_ids, word2idx, idx2word, idx_freqs, vocab_size = load_corpus(corpus_file, min_freq=3)

    # --- Sweep 1: Embedding dimension ---
    dim_values = [50, 100, 200, 300]
    res_dim = run_sweep('embed_dim', dim_values, token_ids, vocab_size, idx_freqs, word2idx, idx2word)
    plot_results('embed_dim', dim_values, res_dim)

    # --- Sweep 2: Context window size ---
    win_values = [2, 3, 5, 7, 10]
    res_win = run_sweep('window', win_values, token_ids, vocab_size, idx_freqs, word2idx, idx2word)
    plot_results('window', win_values, res_win)

    # --- Sweep 3: Negative samples ---
    neg_values = [3, 5, 10, 15, 20]
    res_neg = run_sweep('neg_samples', neg_values, token_ids, vocab_size, idx_freqs, word2idx, idx2word)
    plot_results('neg_samples', neg_values, res_neg)

    print("\nAll hyperparameter sweeps done.")
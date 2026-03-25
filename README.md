# Assignment 2 - Running Guide

This is the guide to run the code for the Programming Assignment - 2 for the course Natural Language Understanding

## Corpus Preparation (Optional Pipeline)

Clean PDFs from dataset/ (writes iitj_corpus_clean.txt, dataset_stats.txt, wordcloud.png):

```bash
python prepare_corpus.py
```

Note: prepare_corpus.py runs immediately on import, so run it only as a script.

## Word2Vec Training

The Word2Vec scripts expect a plain text corpus file named corpus.txt in this folder.
If you want to use iitj_corpus_clean.txt instead, either copy/rename it or update the script paths.

Train skip-gram and CBOW (writes model_*.pkl files):

```bash
python train_word2vec.py
```

Hyperparameter sweeps (writes sweep_*.png and model_*.pkl):

```bash
python hyperparam_tuning.py
```

## Word2Vec Analysis and Outputs

Top-10 words by frequency (writes top_10_words.txt):

```bash
python top_10_words.py
```

Print one word embedding (writes word_embedd.txt and prints to terminal):

1. Edit TARGET_WORD in print_word_embedd.py
2. Ensure model_skipgram_embed_dim_300.pkl exists in this folder
3. Run:

```bash
python print_word_embedd.py
```

Nearest neighbors and analogies:

```bash
python semantic_analysis.py [skipgram_model.pkl] [cbow_model.pkl]
```

PCA / t-SNE visualization:

```bash
python visualize.py [skipgram_model.pkl] [cbow_model.pkl]
```

## RNN Name Generation

Train models:

```bash
python model_rnn.py
python model_blstm.py
python model_rnn_att.py
```

Generate names from trained checkpoints (writes generated_names_*.txt):

```bash
python generate_names.py
```

Evaluate novelty/diversity (reads generated_names_*.txt):

```bash
python evaluate.py
```

Report vanilla RNN params and model size in MB (uses checkpoints/rnn.pt):

```bash
python rnn_model_info.py
```

## Outputs (Common Files)

Word2Vec:
- model_*.pkl
- word_embedd.txt
- top_10_words.txt
- viz_*.png, sweep_*.png

RNN:
- checkpoints/rnn.pt, checkpoints/blstm.pt, checkpoints/attn_rnn.pt
- generated_names_rnn.txt, generated_names_blstm.txt, generated_names_attn.txt
- performance metrics in terminal via evaluate.py

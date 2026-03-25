import numpy as np


class Word2Vec:
    def __init__(self, vocab_size, embed_dim, lr=0.025):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr
        # center word embeddings and context word embeddings
        self.W_in = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
        self.W_out = np.zeros((vocab_size, embed_dim))

    def sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def train_skipgram(self, center_idx, context_idx, neg_indices):
        # grab vectors
        v_c = self.W_in[center_idx]        # center embedding
        u_o = self.W_out[context_idx]       # true context embedding
        u_neg = self.W_out[neg_indices]     # negative sample embeddings

        # forward pass
        pos_score = self.sigmoid(u_o @ v_c)
        neg_scores = self.sigmoid(-u_neg @ v_c)

        # loss = -log(pos) - sum(log(neg))
        loss = -np.log(pos_score + 1e-10) - np.sum(np.log(neg_scores + 1e-10))

        # gradients
        grad_vc = (pos_score - 1) * u_o  # from positive sample
        grad_uo = (pos_score - 1) * v_c

        # negative samples contribution
        for i, nidx in enumerate(neg_indices):
            coeff = (1 - neg_scores[i])
            grad_vc += coeff * u_neg[i]
            self.W_out[nidx] -= self.lr * coeff * v_c

        # update
        self.W_in[center_idx] -= self.lr * grad_vc
        self.W_out[context_idx] -= self.lr * grad_uo

        return loss

    def train_cbow(self, context_indices, center_idx, neg_indices):
        # average context vectors -> v_hat
        v_hat = np.mean(self.W_in[context_indices], axis=0)

        u_o = self.W_out[center_idx]
        u_neg = self.W_out[neg_indices]

        pos_score = self.sigmoid(u_o @ v_hat)
        neg_scores = self.sigmoid(-u_neg @ v_hat)

        loss = -np.log(pos_score + 1e-10) - np.sum(np.log(neg_scores + 1e-10))

        # grad w.r.t. v_hat
        grad_vhat = (pos_score - 1) * u_o
        grad_uo = (pos_score - 1) * v_hat

        for i, nidx in enumerate(neg_indices):
            coeff = (1 - neg_scores[i])
            grad_vhat += coeff * u_neg[i]
            self.W_out[nidx] -= self.lr * coeff * v_hat

        # distribute gradient equally among context words
        grad_per_ctx = self.lr * grad_vhat / len(context_indices)
        for idx in context_indices:
            self.W_in[idx] -= grad_per_ctx

        self.W_out[center_idx] -= self.lr * grad_uo

        return loss

    def get_embeddings(self):
        return self.W_in.copy()


class NegativeSampler:
    # builds the noise distribution P(w) = freq(w)^0.75 / Z
    def __init__(self, word_freqs, vocab_size):
        powered = np.array([word_freqs.get(i, 1) for i in range(vocab_size)], dtype=np.float64)
        powered = np.power(powered, 0.75)
        self.probs = powered / powered.sum()

    def sample(self, k, exclude=None):
        negs = []
        while len(negs) < k:
            candidates = np.random.choice(len(self.probs), size=k * 2, p=self.probs)
            for c in candidates:
                if exclude is not None and c in exclude:
                    continue
                negs.append(c)
                if len(negs) == k:
                    break
        return np.array(negs)
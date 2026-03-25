import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(itos)}
        self.pad = self.stoi["<pad>"]
        self.bos = self.stoi["<s>"]
        self.eos = self.stoi["</s>"]

    @property
    def size(self) -> int:
        return len(self.itos)


class VanillaRNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.w_ih = nn.Linear(emb_dim, hidden_size, bias=True)
        self.w_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, x: torch.Tensor):
        b, t = x.shape
        e = self.emb(x)
        h = torch.zeros(b, self.w_hh.in_features, device=x.device)
        hs = []
        for i in range(t):
            h = torch.tanh(self.w_ih(e[:, i]) + self.w_hh(h))
            hs.append(h)
        h_seq = torch.stack(hs, dim=1)
        return self.proj(h_seq)


class LSTMCell(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.w_ih = nn.Linear(in_dim, 4 * hidden, bias=True)
        self.w_hh = nn.Linear(hidden, 4 * hidden, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        g = self.w_ih(x) + self.w_hh(h)
        i, f, o, u = g.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        c = f * c + i * u
        h = o * torch.tanh(c)
        return h, c


class BLSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fwd = LSTMCell(emb_dim, hidden)
        self.bwd = LSTMCell(emb_dim, hidden)
        self.proj = nn.Linear(2 * hidden, vocab_size, bias=True)

    def forward(self, x: torch.Tensor):
        b, t = x.shape
        e = self.emb(x)

        hf = torch.zeros(b, self.hidden, device=x.device)
        cf = torch.zeros(b, self.hidden, device=x.device)
        fwd_hs = []
        for i in range(t):
            hf, cf = self.fwd(e[:, i], hf, cf)
            fwd_hs.append(hf)
        fwd_seq = torch.stack(fwd_hs, dim=1)

        hb = torch.zeros(b, self.hidden, device=x.device)
        cb = torch.zeros(b, self.hidden, device=x.device)
        bwd_hs = [None] * t
        for i in range(t - 1, -1, -1):
            hb, cb = self.bwd(e[:, i], hb, cb)
            bwd_hs[i] = hb
        bwd_seq = torch.stack(bwd_hs, dim=1)

        hcat = torch.cat([fwd_seq, bwd_seq], dim=-1)
        return self.proj(hcat)


class AttnRNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.w_ih = nn.Linear(emb_dim, hidden, bias=True)
        self.w_hh = nn.Linear(hidden, hidden, bias=False)

        self.attn_q = nn.Linear(hidden, hidden, bias=False)
        self.attn_k = nn.Linear(hidden, hidden, bias=False)
        self.attn_v = nn.Linear(hidden, hidden, bias=False)

        self.mix = nn.Linear(2 * hidden, hidden, bias=True)
        self.proj = nn.Linear(hidden, vocab_size, bias=True)

    def forward(self, x: torch.Tensor):
        b, t = x.shape
        e = self.emb(x)

        h = torch.zeros(b, self.hidden, device=x.device)
        hs = []
        logits_out = []

        for i in range(t):
            h = torch.tanh(self.w_ih(e[:, i]) + self.w_hh(h))
            hs.append(h)

            mem = torch.stack(hs, dim=1)
            q = self.attn_q(h).unsqueeze(1)
            k = self.attn_k(mem)
            v = self.attn_v(mem)

            scores = (q * k).sum(dim=-1) / (self.hidden ** 0.5)
            w = F.softmax(scores, dim=-1).unsqueeze(-1)
            ctx = (w * v).sum(dim=1)

            h2 = torch.tanh(self.mix(torch.cat([h, ctx], dim=-1)))
            logits_out.append(self.proj(h2))

        return torch.stack(logits_out, dim=1)


@torch.no_grad()
def sample_rnn(model: VanillaRNNLM, vocab: Vocab, device, max_len, temperature, top_k):
    model.eval()
    cur = [vocab.bos]
    for _ in range(max_len):
        x = torch.tensor([cur], dtype=torch.long, device=device)
        logits = model(x)
        last = logits[0, -1, :] / max(temperature, 1e-6)

        if top_k and top_k > 0:
            v, ix = torch.topk(last, k=min(top_k, last.size(-1)))
            probs = F.softmax(v, dim=-1)
            pick = ix[torch.multinomial(probs, 1)]
        else:
            probs = F.softmax(last, dim=-1)
            pick = torch.multinomial(probs, 1)

        tok = int(pick.item())
        if tok == vocab.eos:
            break
        cur.append(tok)

    return "".join(vocab.itos[t] for t in cur if t not in (vocab.bos, vocab.eos, vocab.pad))


@torch.no_grad()
def sample_blstm(model: BLSTMLM, vocab: Vocab, device, future_window, max_len, temperature, top_k, min_len=2):
    model.eval()
    seq = [vocab.bos]

    for step in range(max_len):
        tmp = seq + [vocab.pad] * future_window
        x = torch.tensor([tmp], dtype=torch.long, device=device)
        logits = model(x)
        last = logits[0, len(seq) - 1, :] / max(temperature, 1e-6)

        last[vocab.pad] = -1e9
        if step < min_len:
            last[vocab.eos] = -1e9

        if top_k and top_k > 0:
            v, ix = torch.topk(last, k=min(top_k, last.size(-1)))
            probs = F.softmax(v, dim=-1)
            pick = ix[torch.multinomial(probs, 1)]
        else:
            probs = F.softmax(last, dim=-1)
            pick = torch.multinomial(probs, 1)

        tok = int(pick.item())
        if tok == vocab.eos:
            break
        seq.append(tok)

    return "".join(vocab.itos[t] for t in seq if t not in (vocab.bos, vocab.eos, vocab.pad))


@torch.no_grad()
def sample_attn(model: AttnRNNLM, vocab: Vocab, device, max_len, temperature, top_k):
    model.eval()
    cur = [vocab.bos]
    for _ in range(max_len):
        x = torch.tensor([cur], dtype=torch.long, device=device)
        logits = model(x)
        last = logits[0, -1, :] / max(temperature, 1e-6)

        if top_k and top_k > 0:
            v, ix = torch.topk(last, k=min(top_k, last.size(-1)))
            probs = F.softmax(v, dim=-1)
            pick = ix[torch.multinomial(probs, 1)]
        else:
            probs = F.softmax(last, dim=-1)
            pick = torch.multinomial(probs, 1)

        tok = int(pick.item())
        if tok == vocab.eos:
            break
        cur.append(tok)

    return "".join(vocab.itos[t] for t in cur if t not in (vocab.bos, vocab.eos, vocab.pad))


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--max_len", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--future_window", type=int, default=16)

    p.add_argument("--rnn_ckpt", type=str, default="checkpoints/rnn.pt")
    p.add_argument("--blstm_ckpt", type=str, default="checkpoints/blstm.pt")
    p.add_argument("--attn_ckpt", type=str, default="checkpoints/attn_rnn.pt")

    p.add_argument("--out_rnn", type=str, default="generated_names_rnn.txt")
    p.add_argument("--out_blstm", type=str, default="generated_names_blstm.txt")
    p.add_argument("--out_attn", type=str, default="generated_names_attn.txt")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_ckpt(path: str):
        return torch.load(path, map_location=device)

    os.makedirs("checkpoints", exist_ok=True)

    rnn_ckpt = load_ckpt(args.rnn_ckpt)
    vocab = Vocab(rnn_ckpt["vocab_itos"])
    rnn = VanillaRNNLM(vocab.size, rnn_ckpt["hparams"]["emb_dim"], rnn_ckpt["hparams"]["hidden"]).to(device)
    rnn.load_state_dict(rnn_ckpt["state_dict"])

    blstm_ckpt = load_ckpt(args.blstm_ckpt)
    vocab2 = Vocab(blstm_ckpt["vocab_itos"])
    blstm = BLSTMLM(vocab2.size, blstm_ckpt["hparams"]["emb_dim"], blstm_ckpt["hparams"]["hidden"]).to(device)
    blstm.load_state_dict(blstm_ckpt["state_dict"])

    attn_ckpt = load_ckpt(args.attn_ckpt)
    vocab3 = Vocab(attn_ckpt["vocab_itos"])
    attn = AttnRNNLM(vocab3.size, attn_ckpt["hparams"]["emb_dim"], attn_ckpt["hparams"]["hidden"]).to(device)
    attn.load_state_dict(attn_ckpt["state_dict"])

    rnn_names = [
        sample_rnn(rnn, vocab, device, args.max_len, args.temperature, args.top_k) for _ in range(args.n)
    ]
    blstm_names = [
        sample_blstm(blstm, vocab2, device, args.future_window, args.max_len, args.temperature, args.top_k)
        for _ in range(args.n)
    ]
    attn_names = [
        sample_attn(attn, vocab3, device, args.max_len, args.temperature, args.top_k) for _ in range(args.n)
    ]

    with open(args.out_rnn, "w", encoding="utf-8") as f:
        for s in rnn_names:
            f.write(s + "\n")

    with open(args.out_blstm, "w", encoding="utf-8") as f:
        for s in blstm_names:
            f.write(s + "\n")

    with open(args.out_attn, "w", encoding="utf-8") as f:
        for s in attn_names:
            f.write(s + "\n")

    print("wrote:", args.out_rnn, args.out_blstm, args.out_attn)


if __name__ == "__main__":
    main()
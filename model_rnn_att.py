import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_names(path: str) -> list[str]:
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().lower()
            if s:
                names.append(s)
    return names


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad: int
    bos: int
    eos: int

    @property
    def size(self) -> int:
        return len(self.itos)


def build_vocab(names: list[str]) -> Vocab:
    specials = ["<pad>", "<s>", "</s>"]
    chars = sorted(set("".join(names)))
    itos = specials + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad=stoi["<pad>"], bos=stoi["<s>"], eos=stoi["</s>"])


def encode_name(name: str, vocab: Vocab) -> list[int]:
    return [vocab.bos] + [vocab.stoi[c] for c in name] + [vocab.eos]


def pad_batch(seqs: list[list[int]], pad_id: int, device: torch.device):
    lengths = torch.tensor([len(s) for s in seqs], device=device)
    max_len = int(lengths.max().item())
    x = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return x, lengths


class AttnRNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int):
        super().__init__()
        self.vocab_size = vocab_size
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

            scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden)
            w = F.softmax(scores, dim=-1).unsqueeze(-1)
            ctx = (w * v).sum(dim=1)

            h2 = torch.tanh(self.mix(torch.cat([h, ctx], dim=-1)))
            logits_out.append(self.proj(h2))

        logits = torch.stack(logits_out, dim=1)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_batches(encoded: list[list[int]], batch_size: int, shuffle: bool = True):
    idx = list(range(len(encoded)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [encoded[j] for j in idx[i : i + batch_size]]


def loss_on_batch(model: nn.Module, batch_x: torch.Tensor, pad_id: int):
    inp = batch_x[:, :-1]
    tgt = batch_x[:, 1:]
    logits = model(inp)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=pad_id,
    )
    return loss


@torch.no_grad()
def sample_name(
    model: AttnRNNLM,
    vocab: Vocab,
    device: torch.device,
    max_len: int = 24,
    temperature: float = 1.0,
    top_k: int = 0,
):
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

        token = int(pick.item())
        if token == vocab.eos:
            break
        cur.append(token)

    chars = []
    for tok in cur:
        if tok in (vocab.bos, vocab.eos, vocab.pad):
            continue
        chars.append(vocab.itos[tok])
    return "".join(chars)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="TrainingNames.txt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save", type=str, default="checkpoints/attn_rnn.pt")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names = read_names(args.data)
    vocab = build_vocab(names)
    encoded = [encode_name(n, vocab) for n in names]

    model = AttnRNNLM(vocab.size, args.emb_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch_seqs in make_batches(encoded, args.batch_size, shuffle=True):
            bx, _ = pad_batch(batch_seqs, vocab.pad, device)
            opt.zero_grad(set_to_none=True)
            loss = loss_on_batch(model, bx, vocab.pad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            steps += 1

        avg = total_loss / max(steps, 1)
        ppl = math.exp(min(avg, 20))
        print(f"epoch {ep:02d} | loss {avg:.4f} | ppl {ppl:.2f}")

        if ep in (1, args.epochs // 2, args.epochs):
            s = sample_name(model, vocab, device)
            print("sample:", s)

    ckpt = {
        "model": "attn_rnn",
        "state_dict": model.state_dict(),
        "vocab_itos": vocab.itos,
        "hparams": {
            "emb_dim": args.emb_dim,
            "hidden": args.hidden,
        },
        "num_params": model.num_params(),
    }
    torch.save(ckpt, args.save)
    print("saved:", args.save)
    print("trainable_params:", ckpt["num_params"])


if __name__ == "__main__":
    main()
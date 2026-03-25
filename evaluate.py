def read_set(path: str) -> set[str]:
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().lower()
            if s:
                out.add(s)
    return out


def read_list(path: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().lower()
            if s:
                out.append(s)
    return out


def novelty_rate(generated: list[str], train_set: set[str]) -> float:
    if not generated:
        return 0.0
    novel = sum(1 for s in generated if s not in train_set)
    return 100.0 * novel / len(generated)


def diversity(generated: list[str]) -> float:
    if not generated:
        return 0.0
    return len(set(generated)) / len(generated)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="TrainingNames.txt")
    p.add_argument("--rnn", type=str, default="generated_names_rnn.txt")
    p.add_argument("--blstm", type=str, default="generated_names_blstm.txt")
    p.add_argument("--attn", type=str, default="generated_names_attn.txt")
    args = p.parse_args()

    train = read_set(args.train)

    def report(tag: str, path: str):
        gen = read_list(path)
        nov = novelty_rate(gen, train)
        div = diversity(gen)
        print(f"{tag:8s} | total {len(gen):4d} | novelty {nov:6.2f}% | diversity {div:6.3f}")

    report("rnn", args.rnn)
    report("blstm", args.blstm)
    report("attn", args.attn)


if __name__ == "__main__":
    main()
from collections import Counter
import os


def load_tokens(corpus_path):
	with open(corpus_path, "r", encoding="utf-8") as f:
		text = f.read()
	return text.lower().split()


def main():
	base_dir = os.path.dirname(__file__)
	corpus_path = os.path.join(base_dir, "corpus.txt")
	output_path = os.path.join(base_dir, "top_10_words.txt")

	tokens = load_tokens(corpus_path)
	counts = Counter(tokens)
	top_10 = counts.most_common(10)

	parts = []
	for word, freq in top_10:
		parts.append(f"{word}")
		parts.append(f"{freq}")

	line = ", ".join(parts)
	with open(output_path, "w", encoding="utf-8") as f:
		f.write(line + "\n")
	print(line)


if __name__ == "__main__":
	main()

import os
import pickle


TARGET_WORD = "lagrangian"


def load_model(model_path):
	with open(model_path, "rb") as f:
		data = pickle.load(f)
	return data


def format_embedding(vec, decimals=4):
	return ", ".join(f"{v:.{decimals}f}" for v in vec)


def main():
	base_dir = os.path.dirname(__file__)
	model_path = os.path.join(base_dir, "model_skipgram_embed_dim_300.pkl")
	output_path = os.path.join(base_dir, "word_embedd.txt")

	data = load_model(model_path)
	embeddings = data["W_in"]
	word2idx = data["word2idx"]

	word = TARGET_WORD.lower()
	if word not in word2idx:
		raise ValueError(f"'{TARGET_WORD}' not in vocabulary. Choose another word.")

	vec = embeddings[word2idx[word]]
	line = f"{word} - {format_embedding(vec)}"

	with open(output_path, "w", encoding="utf-8") as f:
		f.write(line + "\n")

	print(line)


if __name__ == "__main__":
	main()

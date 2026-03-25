import argparse
import os

import torch


def get_state_dict(ckpt):
	if isinstance(ckpt, dict) and "state_dict" in ckpt:
		return ckpt["state_dict"]
	return ckpt


def count_params_and_bytes(state_dict):
	num_params = 0
	num_bytes = 0
	for t in state_dict.values():
		if hasattr(t, "numel"):
			num_params += t.numel()
			num_bytes += t.numel() * t.element_size()
	return num_params, num_bytes


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--ckpt",
		type=str,
		default=os.path.join("checkpoints", "rnn.pt"),
		help="Path to the saved vanilla RNN checkpoint",
	)
	args = parser.parse_args()

	base_dir = os.path.dirname(__file__)
	ckpt_path = os.path.join(base_dir, args.ckpt)

	ckpt = torch.load(ckpt_path, map_location="cpu")
	state_dict = get_state_dict(ckpt)

	num_params, num_bytes = count_params_and_bytes(state_dict)
	model_size_mb = num_bytes / (1024 * 1024)

	print(f"num_params: {num_params}")
	print(f"model_size_mb: {model_size_mb:.4f}")


if __name__ == "__main__":
	main()

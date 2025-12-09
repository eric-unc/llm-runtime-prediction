import sqlite3
import numpy as np
import re
from pathlib import Path
from statistics import median
from transformers import AutoTokenizer

def token_count(text: str, tokenizer) -> int:
	return len(tokenizer.encode(text, add_special_tokens=False))

def print_pearson(conn, model, full_model_path):
	tokenizer = AutoTokenizer.from_pretrained(full_model_path)

	rows = conn.execute(
		"SELECT prompt, response, duration FROM results WHERE model=?",
		(model,),
	).fetchall()

	if not rows:
		print(f"No data for {model}")
		return
	
	# group durations by prompt
	durations_by_prompt = {}
	responses_by_prompt = {}
	for prompt, response, duration in rows:
		durations_by_prompt.setdefault(prompt, []).append(duration)
		responses_by_prompt[prompt] = response

	# compute median runtime per prompt
	medians = []
	prompt_lens = []
	response_lens = []

	for prompt, durs in durations_by_prompt.items():
		medians.append(median(durs))
		prompt_lens.append(token_count(prompt, tokenizer))
		response_lens.append(token_count(responses_by_prompt[prompt], tokenizer))

	medians = np.array(medians)
	prompt_lens = np.array(prompt_lens)
	response_lens = np.array(response_lens)

	def pearson(x, y):
		if len(x) < 2:
			return np.nan
		return np.corrcoef(x, y)[0, 1]

	corr_prompt_time = pearson(prompt_lens, medians)
	corr_response_time = pearson(response_lens, medians)

	print(f"\nModel: {model}")
	print(f"Samples: {len(medians)}")
	print(f"Prompt length ↔ Median runtime:  {corr_prompt_time:.5f}")
	print(f"Response length ↔ Median runtime: {corr_response_time:.5f}")


if __name__ == "__main__":
	conn = sqlite3.connect(Path("output/alpaca.db"))

	print_pearson(conn, "llama3.2", "meta-llama/Llama-3.2-3B")
	print_pearson(conn, "llama3.3", "meta-llama/Llama-3.3-70B-Instruct")

	conn.close()


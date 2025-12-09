import sqlite3
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import median
from transformers import AutoTokenizer

def token_count(text: str, tokenizer) -> int:
	return len(tokenizer.encode(text, add_special_tokens=False))

def load_median_data(conn, model, tokenizer):
	rows = conn.execute(
		"SELECT prompt, response, duration FROM results WHERE model=?",
		(model,),
	).fetchall()
	if not rows:
		return np.array([]), np.array([]), np.array([])

	durations_by_prompt = {}
	responses_by_prompt = {}
	for prompt, response, duration in rows:
		durations_by_prompt.setdefault(prompt, []).append(duration)
		responses_by_prompt[prompt] = response

	medians, prompt_lens, response_lens = [], [], []
	for prompt, durs in durations_by_prompt.items():
		medians.append(median(durs))
		prompt_lens.append(token_count(prompt, tokenizer))
		response_lens.append(token_count(responses_by_prompt[prompt], tokenizer))

	# Filter out responses longer than 10k tokens
	mask = np.array(response_lens) <= 10_000
	prompt_lens, response_lens, medians = np.array(prompt_lens)[mask], np.array(response_lens)[mask], np.array(medians)[mask]

	return np.array(prompt_lens), np.array(response_lens), np.array(medians)

def plot_scatter(x, y, xlabel, ylabel, title, filename, fit_line=False):
	plt.figure(figsize=(6, 4))
	plt.scatter(x, y, alpha=0.6)
	if fit_line and len(x) > 1:
		slope, intercept = np.polyfit(x, y, 1)
		x_fit = np.linspace(min(x), max(x), 100)
		y_fit = slope * x_fit + intercept
		plt.plot(x_fit, y_fit, color="red", label="Best fit")
		plt.legend()
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.tight_layout()
	out_path = Path(filename)
	plt.savefig(out_path)
	plt.close()
	print(f"Saved: {out_path}")

def plot_model(conn, model, full_model_path):
	tokenizer = AutoTokenizer.from_pretrained(full_model_path)
	prompt_lens, response_lens, medians = load_median_data(conn, model, tokenizer)

	# Plot 1: prompt length vs runtime
	plot_scatter(
		prompt_lens,
		medians,
		"Prompt length (tokens)",
		"Median runtime (s)",
		f"{model}: Prompt length vs runtime",
		f"output/{model}_prompt_vs_runtime.png",
	)

	# Plot 2: output length vs runtime (with best-fit line)
	plot_scatter(
		response_lens,
		medians,
		"Output length (tokens)",
		"Median runtime (s)",
		f"{model}: Output length vs runtime",
		f"output/{model}_output_vs_runtime.png",
		fit_line=True,
	)

if __name__ == "__main__":
	conn = sqlite3.connect(Path("output/alpaca.db"))

	plot_model(conn, "llama3.2", "meta-llama/Llama-3.2-3B")
	plot_model(conn, "llama3.3", "meta-llama/Llama-3.3-70B-Instruct")

	conn.close()

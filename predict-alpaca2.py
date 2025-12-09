import json
import subprocess
import argparse

with open("input/alpaca_data_cleaned.json", "r") as f:
	queries = json.load(f)

if not isinstance(queries, list):
	raise ValueError("Unexpected format...")

len_queries = len(queries)

models = [
	#"llama3.2", # 3b
	"llama32-ft", # 3b ish? idk
	]
runs = 10

for model in models:
	if model != "llama32-ft":
		# Make sure the model is pulled first...
		subprocess.run([
			"ollama",
			"pull",
			model,
		], check=True)

	for idx, query in enumerate(queries, start=1):
		if idx > 1500 + 1:
			break

		inst = query['instruction']
		inp = query['input']

		# see https://github.com/tatsu-lab/stanford_alpaca
		if len(inp) == 0:
			prompt = (
				"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
				"### Instruction:\n"
				f"{inst}\n\n"
				"### Response:\n"
			)
		else:
			prompt = (
				"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
				"### Instruction:\n"
				f"{inst}\n\n"
				"### Input:\n"
				f"{inp}\n\n"
				"### Response:\n"
			)

		# We want to estimate the prompt
		est_prompt = (
			"You are given a prompt. Do not answer it. Predict the output token count that the Llama 3.3 LLM will produce when answering the prompt. Only output a number.\n\n"
			"## Prompt:\n"
			"```\n"
			f"{prompt}\n"
			"```\n\n"
			"## Output token count:\n"
		)
		
		print(f"\n[{model}] Prompt {idx}/{len_queries}: {est_prompt[:200]}... ---")
		subprocess.run([
			"python3",
			"bench-prompt.py",
			model,
			est_prompt,
			"output/alpaca-pred2.db",
			"--runs", str(runs)
		], check=True)


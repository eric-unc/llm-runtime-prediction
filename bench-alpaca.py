import json
import subprocess
import argparse

with open("input/alpaca_data_cleaned.json", "r") as f:
	queries = json.load(f)

if not isinstance(queries, list):
	raise ValueError("Unexpected format...")

len_queries = len(queries)

models = ["llama3.2", "llama3.3"]
runs = 10
#models = ["gpt-oss"]
#runs = 1

for model in models:
	# Make sure the model is pulled first...
	subprocess.run([
		"ollama",
		"pull",
		model,
	], check=True)

	for idx, query in enumerate(queries, start=1):
		# Processing all of this would take a long time, so we're just gonna do the first 1000 for now
		if idx < 1000:
			continue
		if idx > 1000 + 1:
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
		
		print(f"\n[{model}] Prompt {idx}/{len_queries}: {prompt[:200]}... ---")
		subprocess.run([
			"python3",
			"bench-prompt.py",
			model,
			prompt,
			"output/alpaca.db",
			"--runs", str(runs)
		], check=True)


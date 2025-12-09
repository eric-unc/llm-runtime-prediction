import sqlite3
import numpy as np
from pathlib import Path
from statistics import median

def print_median(conn, model):
	rows = conn.execute(
		"SELECT prompt, duration FROM results WHERE model=?",
		(model,),
	).fetchall()

	if not rows:
		print(f"No data for {model}")
		return
	
	# group durations by prompt
	durations_by_prompt = {}
	for prompt, duration in rows:
		durations_by_prompt.setdefault(prompt, []).append(duration)

	# compute median runtime per prompt
	medians = {}
	for prompt, durs in durations_by_prompt.items():
		medians.setdefault(prompt, []).append(median(durs))

	for prompt, m in medians.items():
		print(prompt, "-", m)

if __name__ == "__main__":
	conn = sqlite3.connect(Path("output/example.db"))

	print_median(conn, "llama3.3")

	conn.close()
	

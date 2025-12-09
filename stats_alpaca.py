import sqlite3
import numpy as np
from collections import defaultdict

DB_PATH = "output/alpaca.db"

def load_original_token_counts(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT model, response FROM results WHERE run_index = 0")

    # model -> list of token counts
    counts = defaultdict(list)

    for model, response in cur.fetchall():
        # Define “token count” here; adjust if needed.
        tok_count = len(response.split())
        counts[model].append(tok_count)

    conn.close()
    return counts


def compute_stats(counts_per_model):
    stats = {}
    for model, counts in counts_per_model.items():
        if counts:
            arr = np.array(counts)
            stats[model] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "num_samples": len(arr),
            }
        else:
            stats[model] = {
                "mean": 0.0,
                "std": 0.0,
                "num_samples": 0,
            }
    return stats


if __name__ == "__main__":
    counts = load_original_token_counts()
    stats = compute_stats(counts)

    print("Per-model token count statistics:")
    for model, s in stats.items():
        print(f"{model}: mean={s['mean']:.3f}, std={s['std']:.3f}, samples={s['num_samples']}")

    # Global stats across all models
    all_counts = []
    for model in counts:
        all_counts.extend(counts[model])

    if all_counts:
        arr = np.array(all_counts)
        print("\nGlobal statistics:")
        print(f"mean={arr.mean():.3f}, std={arr.std():.3f}, samples={len(arr)}")
    else:
        print("No data found.")


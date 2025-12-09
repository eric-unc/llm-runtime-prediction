import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer

def token_count(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

# ----------------------------------------------------------------------
# User-supplied mapping: original_prompt -> prediction_prompt
# Fill in your own logic here.
# ----------------------------------------------------------------------
def map_original_to_prediction_prompt(original_prompt: str) -> str:
    est_prompt = (
        "You are given a prompt. Do not answer it. Predict the output token count that the Llama 3.3 LLM will produce when answering the prompt. Only output a number.\n\n"
        "## Prompt:\n"
        "```\n"
        f"{original_prompt}\n"
        "```\n\n"
        "## Output token count:\n"
    )
    return est_prompt


# ----------------------------------------------------------------------
# Load original outputs: (prompt -> true_output_token_count)
# Each prompt may have several runs; assume true token count is length of response.
# ----------------------------------------------------------------------
def load_training_results(path="output/alpaca.db"):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    # Load before Dec 7, for the first 1000 prompts
    cur.execute("SELECT prompt, response FROM results WHERE model = 'llama3.3' AND timestamp < '2025-12-06 23:59:59'")
    data = defaultdict(list)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    for prompt, response in cur.fetchall():
        data[prompt].append(token_count(response, tokenizer))
    conn.close()

    # Use the first run’s token count or median; pick median to be robust.
    return {p: int(np.median(counts)) for p, counts in data.items()}

def load_validation_results(path="output/alpaca.db"):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    # Load On/after Dec 7, for the last ~500 prompts
    cur.execute("SELECT prompt, response FROM results WHERE model = 'llama3.3' AND timestamp > '2025-12-06 23:59:59'")
    data = defaultdict(list)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    for prompt, response in cur.fetchall():
        data[prompt].append(token_count(response, tokenizer))
    conn.close()

    # Use the first run’s token count or median; pick median to be robust.
    return {p: int(np.median(counts)) for p, counts in data.items()}

# ----------------------------------------------------------------------
# Load prediction results:
# (model -> prompt -> list of (predicted_count, runtime))
# Predicted count parsed as int if possible; otherwise None.
# ----------------------------------------------------------------------
def load_prediction_results(path="output/alpaca-pred2.db"):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT model, prompt, response, duration FROM results")

    preds = defaultdict(lambda: defaultdict(list))
    for model, prompt, response, duration in cur.fetchall():
        # parse integer prediction if possible
        try:
            pred = int(response.strip())
        except Exception:
            pred = None
        preds[model][prompt].append((pred, duration))

    conn.close()
    return preds


# ----------------------------------------------------------------------
# Compute metrics for each prediction model.
# ----------------------------------------------------------------------
def compute_metrics(original, predictions):
    models = predictions.keys()

    percent_correct = {}
    percent_correct_by_5 = {}
    percent_correct_by_10 = {}
    percent_correct_by_15 = {}
    percent_correct_by_20 = {}
    percent_correct_by_25 = {}
    median_runtime = {}
    avg_std = {}

    for model in models:
        model_preds = predictions[model]

        # Containers
        correct = 0
        correct_by_5 = 0
        correct_by_10 = 0
        correct_by_15 = 0
        correct_by_20 = 0
        correct_by_25 = 0
        total = 0
        runtimes = []
        stds = []

        for orig_prompt, true_count in original.items():
            pred_prompt = map_original_to_prediction_prompt(orig_prompt)

            if pred_prompt not in model_preds:
                continue

            runs = model_preds[pred_prompt]
            values = [p for (p, _) in runs if p is not None]
            durs = [d for (_, d) in runs]

            if values:
                correct += sum(1 for v in values if v == true_count)
                correct_by_5 += sum(1 for v in values if abs(v - true_count) <= 5)
                correct_by_10 += sum(1 for v in values if abs(v - true_count) <= 10)
                correct_by_15 += sum(1 for v in values if abs(v - true_count) <= 15)
                correct_by_20 += sum(1 for v in values if abs(v - true_count) <= 20)
                correct_by_25 += sum(1 for v in values if abs(v - true_count) <= 25)
                total += len(values)

                # runtime: median per-prompt, then we will average across prompts
                runtimes.append(np.median(durs))

                # std per prompt
                if len(values) > 1:
                    stds.append(np.std(values))
                else:
                    stds.append(0.0)

        percent_correct[model] = (correct / total) if total > 0 else 0.0
        print(f"[{model}] Percent correct: {percent_correct[model]}")
        percent_correct_by_5[model] = (correct_by_5 / total) if total > 0 else 0.0
        print(f"[{model}] Percent correct (<= 5 tokens): {percent_correct_by_5[model]}")
        percent_correct_by_10[model] = (correct_by_10 / total) if total > 0 else 0.0
        print(f"[{model}] Percent correct (<= 10 tokens): {percent_correct_by_10[model]}")
        percent_correct_by_15[model] = (correct_by_15 / total) if total > 0 else 0.0
        print(f"[{model}] Percent correct (<= 15 tokens): {percent_correct_by_15[model]}")
        percent_correct_by_20[model] = (correct_by_20 / total) if total > 0 else 0.0
        print(f"[{model}] Percent correct (<= 20 tokens): {percent_correct_by_20[model]}")
        percent_correct_by_25[model] = (correct_by_25 / total) if total > 0 else 0.0
        print(f"[{model}] Percent correct (<= 25 tokens): {percent_correct_by_25[model]}")
        median_runtime[model] = float(np.median(runtimes)) if runtimes else 0.0
        print(f"[{model}] Median runtime: {median_runtime[model]}")
        avg_std[model] = float(np.mean(stds)) if stds else 0.0
        print(f"[{model}] Avg std: {avg_std[model]}")

    return percent_correct, percent_correct_by_5, percent_correct_by_10, percent_correct_by_15, percent_correct_by_20, percent_correct_by_25, median_runtime, avg_std


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def plot_bars(title, out_path, metric_dict, ylabel):
    models = list(metric_dict.keys())
    values = [metric_dict[m] for m in models]

    plt.figure(figsize=(8, 4))
    plt.bar(models, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(Path(out_path))
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    original1 = load_training_results()
    original2 = load_validation_results()
    predictions = load_prediction_results()

    print("Training:")
    pct_correct, pct_correct_5, pct_correct_10, pct_correct_15, pct_correct_20, pct_correct_25, med_runtime, avg_std = compute_metrics(original1, predictions)

    plot_bars(
        "Percentage of Correct Predictions (training)",
        "output/llama3.3_pred_training_percentages.png",
        pct_correct,
        "Percent Correct (training)")

    plot_bars(
        "Percentage of Correct Predictions (within 5 tokens) (training)",
        "output/llama3.3_pred_training_percentages_5.png",
        pct_correct_5,
        "Percent Correct (within 5 tokens) (training)")

    plot_bars(
        "Percentage of Correct Predictions (within 10 tokens) (training)",
        "output/llama3.3_pred_training_percentages_10.png",
        pct_correct_10,
        "Percent Correct (within 10 tokens) (training)")

    plot_bars(
        "Percentage of Correct Predictions (within 15 tokens) (training)",
        "output/llama3.3_pred_training_percentages_15.png",
        pct_correct_15,
        "Percent Correct (within 15 tokens) (training)")

    plot_bars(
        "Percentage of Correct Predictions (within 20 tokens) (training)",
        "output/llama3.3_pred_training_percentages_20.png",
        pct_correct_20,
        "Percent Correct (within 20 tokens) (training)")

    plot_bars(
        "Percentage of Correct Predictions (within 25 tokens) (training)",
        "output/llama3.3_pred_training_percentages_25.png",
        pct_correct_25,
        "Percent Correct (within 25 tokens) (training)")

    plot_bars(
        "Median Runtime per Model (training)",
        "output/llama3.3_pred_training_medians.png",
        med_runtime,
        "Median Runtime (s) (training)")

    plot_bars(
        "Average Std Dev Across Runs per Prompt (training)",
        "output/llama3.3_pred_training_stddev.png",
        avg_std,
        "Std Dev (training)")

    print("\nValidation:")
    pct_correct, pct_correct_5, pct_correct_10, pct_correct_15, pct_correct_20, pct_correct_25, med_runtime, avg_std = compute_metrics(original2, predictions)

    plot_bars(
        "Percentage of Correct Predictions (validation)",
        "output/llama3.3_pred_validation_percentages.png",
        pct_correct,
        "Percent Correct (validation)")

    plot_bars(
        "Percentage of Correct Predictions (within 5 tokens) (validation)",
        "output/llama3.3_pred_validation_percentages_5.png",
        pct_correct_5,
        "Percent Correct (within 5 tokens) (training)")

    plot_bars(
        "Percentage of Correct Predictions (within 10 tokens) (validation)",
        "output/llama3.3_pred_validation_percentages_10.png",
        pct_correct_10,
        "Percent Correct (within 10 tokens) (training)")

    plot_bars(
        "Percentage of Correct Predictions (within 15 tokens) (validation)",
        "output/llama3.3_pred_validation_percentages_15.png",
        pct_correct_15,
        "Percent Correct (within 15 tokens) (validation)")

    plot_bars(
        "Percentage of Correct Predictions (within 20 tokens) (validation)",
        "output/llama3.3_pred_validation_percentages_20.png",
        pct_correct_20,
        "Percent Correct (within 20 tokens) (validation)")

    plot_bars(
        "Percentage of Correct Predictions (within 25 tokens) (validation)",
        "output/llama3.3_pred_validation_percentages_25.png",
        pct_correct_25,
        "Percent Correct (within 25 tokens) (validation)")

    plot_bars(
        "Median Runtime per Model (validation)",
        "output/llama3.3_pred_validation_medians.png",
        med_runtime,
        "Median Runtime (s) (validation)")

    plot_bars(
        "Average Std Dev Across Runs per Prompt (training)",
        "output/llama3.3_pred_training_stddev.png",
        avg_std,
        "Std Dev (training)")


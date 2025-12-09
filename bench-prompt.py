import time
import requests
import numpy as np
import argparse
import sqlite3
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"

def query_ollama(model: str, prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "seed": 1,
            "temperature": 0,
            #"num_predict": 10 # TODO: set just for predictions
        },
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def init_db(path: Path):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            run_index INTEGER NOT NULL,
            response TEXT NOT NULL,
            duration REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model, prompt, run_index)
        )
    """)
    conn.commit()
    return conn

def already_ran(conn, model, prompt, runs):
    cur = conn.execute(
        "SELECT COUNT(*) FROM results WHERE model=? AND prompt=?",
        (model, prompt),
    )
    count = cur.fetchone()[0]
    return count >= runs

def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama model inference times.")
    parser.add_argument("model", help="Model name, e.g., deepseek-r1:8b")
    parser.add_argument("prompt", help="Prompt to send")
    parser.add_argument("db", help="Path to output database")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    conn = init_db(args.db)

    if already_ran(conn, args.model, args.prompt, args.runs):
        print("All runs already exist in database. Skipping.")
        return

    # warm-up run
    _ = query_ollama(args.model, "Say 'Hi' and nothing else.")

    for i in range(args.runs):
        cur = conn.execute(
            "SELECT 1 FROM results WHERE model=? AND prompt=? AND run_index=?",
            (args.model, args.prompt, i),
        )
        if cur.fetchone():
            print(f"Skipping run {i+1}, already recorded.")
            continue

        start = time.perf_counter()
        response = query_ollama(args.model, args.prompt)
        end = time.perf_counter()
        duration = end - start

        conn.execute(
            "INSERT OR IGNORE INTO results (model, prompt, run_index, response, duration) VALUES (?, ?, ?, ?, ?)",
            (args.model, args.prompt, i, response.strip(), duration),
        )
        conn.commit()
        print(f"Run {i+1}: {duration:.2f}s")

    # Summary
    cur = conn.execute(
        "SELECT duration FROM results WHERE model=? AND prompt=? ORDER BY run_index",
        (args.model, args.prompt),
    )
    times = [row[0] for row in cur.fetchall()]
    if times:
        avg = np.mean(times)
        std = np.std(times)
        print(f"\nAverage: {avg:.2f}s  StdDev: {std:.2f}s")

    conn.close()

if __name__ == "__main__":
    main()


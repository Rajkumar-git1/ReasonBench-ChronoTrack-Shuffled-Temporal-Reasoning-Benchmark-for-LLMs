"""
ChronoTrack evaluation harness.

Runs the dataset against any OpenAI-compatible /v1/chat/completions endpoint
(this covers Ollama running locally, vLLM, LM Studio, Together AI, etc.),
scores exact-match on the extracted answer, and reports accuracy broken down
by order_mode x n_distractors x query_type, plus a self-consistency check
using the paraphrased question.

USAGE
-----
1. Stand up your models. Easiest path is Ollama:
     ollama pull llama3
     ollama pull mistral
     ollama pull phi3
     ollama pull gemma
   Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1

2. Edit MODEL_CONFIGS below (base_url / model name) if not using Ollama.

3. Run:
     python3 eval_harness.py --dataset chronotrack_dataset.jsonl --limit 60

   --limit caps items per model for a quick pilot run; drop it for the full set.

OUTPUT
------
- results_raw.csv     one row per (model, item) with prediction + correctness
- results_summary.csv accuracy pivoted by model x condition
- printed summary table
"""

import argparse
import csv
import json
import re
import time
import urllib.request
import urllib.error

MODEL_CONFIGS = [
    {"name": "llama3",  "base_url": "http://localhost:11434/v1", "model_id": "llama3"},
    {"name": "mistral", "base_url": "http://localhost:11434/v1", "model_id": "mistral"},
    {"name": "phi3",    "base_url": "http://localhost:11434/v1", "model_id": "phi3"},
    {"name": "gemma",   "base_url": "http://localhost:11434/v1", "model_id": "gemma"},
]

SYSTEM_PROMPT = (
    "You will read a short story describing where objects were moved or who "
    "received them, with explicit days of the week. Answer the question that "
    "follows using information from the story only. "
    "Respond with EXACTLY one line in the form: Answer: <value>\n"
    "The <value> must be a single word or name copied from the story (a place or a person), "
    "with no extra explanation."
)


def call_model(base_url, model_id, story, question, timeout=60):
    prompt = f"Story: {story}\n\nQuestion: {question}"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 30,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    latency = time.time() - t0
    text = data["choices"][0]["message"]["content"]
    return text, latency


def extract_answer(raw_text):
    m = re.search(r"answer\s*:\s*(.+)", raw_text, re.IGNORECASE)
    candidate = m.group(1) if m else raw_text
    candidate = candidate.strip().strip(".").strip()
    # keep first line/token cluster only
    candidate = candidate.split("\n")[0].strip()
    return candidate


def normalize(s):
    return re.sub(r"[^a-z]", "", s.lower()) if s else ""


def run_eval(dataset_path, limit=None):
    with open(dataset_path) as f:
        items = [json.loads(line) for line in f]
    if limit:
        items = items[:limit]

    raw_rows = []
    for cfg in MODEL_CONFIGS:
        print(f"\n=== Evaluating {cfg['name']} ({len(items)} items) ===")
        for i, item in enumerate(items):
            try:
                raw, latency = call_model(cfg["base_url"], cfg["model_id"],
                                           item["story"], item["question"])
                pred = extract_answer(raw)
                correct = normalize(pred) == normalize(item["gold_answer"])
            except (urllib.error.URLError, TimeoutError, KeyError) as e:
                pred, latency, correct = f"ERROR: {e}", None, False

            # self-consistency probe on paraphrase
            try:
                raw2, _ = call_model(cfg["base_url"], cfg["model_id"],
                                      item["story"], item["question_paraphrase"])
                pred2 = extract_answer(raw2)
                self_consistent = normalize(pred2) == normalize(pred)
            except (urllib.error.URLError, TimeoutError, KeyError):
                self_consistent = None

            raw_rows.append({
                "model": cfg["name"],
                "item_id": item["id"],
                "order_mode": item["meta"]["order_mode"],
                "n_distractors": item["meta"]["n_distractors"],
                "query_type": item["meta"]["query_type"],
                "prediction": pred,
                "gold": item["gold_answer"],
                "correct": correct,
                "self_consistent": self_consistent,
                "latency_s": round(latency, 2) if latency else None,
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(items)} done")

    return raw_rows


def summarize(raw_rows, out_prefix="results"):
    with open(f"{out_prefix}_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    # aggregate: accuracy by model x order_mode x n_distractors x query_type
    from collections import defaultdict
    buckets = defaultdict(lambda: [0, 0])  # key -> [correct_count, total_count]
    consistency = defaultdict(lambda: [0, 0])

    for row in raw_rows:
        key = (row["model"], row["order_mode"], row["n_distractors"], row["query_type"])
        buckets[key][1] += 1
        if row["correct"]:
            buckets[key][0] += 1
        if row["self_consistent"] is not None:
            consistency[row["model"]][1] += 1
            if row["self_consistent"]:
                consistency[row["model"]][0] += 1

    with open(f"{out_prefix}_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "order_mode", "n_distractors", "query_type", "accuracy", "n"])
        for (model, order, dist, qtype), (correct, total) in sorted(buckets.items()):
            acc = correct / total if total else 0
            writer.writerow([model, order, dist, qtype, round(acc, 3), total])

    print("\n=== Summary (accuracy by condition) ===")
    for (model, order, dist, qtype), (correct, total) in sorted(buckets.items()):
        acc = correct / total if total else 0
        print(f"{model:10s} order={order:13s} distractors={dist} qtype={qtype:14s} "
              f"acc={acc:.2f} (n={total})")

    print("\n=== Self-consistency (agreement across paraphrased query) ===")
    for model, (correct, total) in consistency.items():
        rate = correct / total if total else 0
        print(f"{model:10s} self-consistency={rate:.2f} (n={total})")

    print("\n=== Key diagnostic: chronological vs scrambled accuracy gap ===")
    per_model_order = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in raw_rows:
        per_model_order[row["model"]][row["order_mode"]][1] += 1
        if row["correct"]:
            per_model_order[row["model"]][row["order_mode"]][0] += 1
    for model, ord_data in per_model_order.items():
        c_correct, c_total = ord_data.get("chronological", [0, 1])
        s_correct, s_total = ord_data.get("scrambled", [0, 1])
        c_acc = c_correct / c_total if c_total else 0
        s_acc = s_correct / s_total if s_total else 0
        print(f"{model:10s} chronological={c_acc:.2f}  scrambled={s_acc:.2f}  "
              f"gap={c_acc - s_acc:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="chronotrack_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                         help="cap items per model for a quick pilot run")
    args = parser.parse_args()

    rows = run_eval(args.dataset, limit=args.limit)
    summarize(rows)

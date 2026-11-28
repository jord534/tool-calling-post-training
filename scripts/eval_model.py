import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk

from tool_calling_post_training.evaluate import parse_answer, score_bfcl, score_calls
from tool_calling_post_training.load import get_model, get_tokenizer, pick_device
from tool_calling_post_training.tools import build_messages, convert_to_xlam_tool, parse_json_field

REPO_ROOT = Path(__file__).resolve().parent.parent


#### CONFIGS
def load_yaml(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find the config file: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def find_model(models_config: dict, name: str) -> dict:
    for model in models_config["models"]:
        if model["name"] == name:
            return model
    names = [m["name"] for m in models_config["models"]]
    raise ValueError(f"No model named {name!r} in the models config, expected one of {names}")


def read_system_message(cache_dir: Path):
    # the system message used in training, so prompts match it exactly
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest in {cache_dir}: run prepare_xlam.py first")
    with open(manifest_path) as f:
        return json.load(f)["system_prompt"]
####


#### TEST SETS
# every loader returns items with the same keys:
# test, category, id, query, tools (parsed list), reference (calls, bfcl ground truth, or None)

def load_xlam(cache_dir: Path, split: str) -> list[dict]:
    # split: "test" for the held-out test, "val" for validation mode
    raw_dir = cache_dir / "raw" / split
    if not raw_dir.exists():
        raise FileNotFoundError(f"No raw rows in {raw_dir}: prepare_xlam.py must save raw/{split}")
    items = []
    for row in load_from_disk(str(raw_dir)):
        items.append({
            "test": f"xlam_{split}",
            "category": "xlam",
            "id": str(row["id"]),
            "query": row["query"],
            "tools": parse_json_field(row["tools"]),
            "reference": parse_json_field(row["answers"]),
        })
    return items


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}: run fetch_benchmarks.py first")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bfcl(bfcl_config: dict) -> list[dict]:
    data_dir = REPO_ROOT / bfcl_config["dir"] / bfcl_config["sparse_path"]
    items = []
    for category in bfcl_config["categories"]:
        questions = read_jsonl(data_dir / f"BFCL_v4_{category}.json")
        # irrelevance has no answer file: the correct answer is "no call"
        answers = {}
        if category != "irrelevance":
            answers = {a["id"]: a["ground_truth"] for a in read_jsonl(data_dir / "possible_answer" / f"BFCL_v4_{category}.json")}
        for q in questions:
            items.append({
                "test": "bfcl",
                "category": category,
                "id": q["id"],
                "query": q["question"][0][0]["content"],  # first turn, the user message
                "tools": q["function"],
                "reference": answers.get(q["id"]) if category != "irrelevance" else None,
            })
    return items


def load_when2call(w2c_config: dict) -> list[dict]:
    rows = read_jsonl(REPO_ROOT / w2c_config["dir"] / w2c_config["test_file"])
    items = []
    for row in rows:
        items.append({
            "test": "when2call",
            "category": row["correct_answer"],  # tool_call, request_for_info or cannot_answer
            "id": row["uuid"],
            "query": row["question"],
            "tools": [parse_json_field(t) for t in row["tools"]],  # a list of json strings, empty for 258 items
            "reference": None,
        })
    return items
####


#### GENERATION
def render_prompts(items: list[dict], tokenizer, system_message) -> list[str]:
    # the same prompt as in training: build_messages, then the chat template up to the answer
    prompts = []
    for item in items:
        messages = build_messages(item["tools"], item["query"], system_message)
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


@torch.no_grad()
def generate_answers(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    # greedy decoding, in batches of similar lengths; returns only the generated text, in the input order
    tokenizer.padding_side = "left"  # left padding: every prompt ends where generation starts
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    answers = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        batch_ids = order[start:start + batch_size]
        # the template already wrote the begin-of-sentence token as text: don't add another one
        inputs = tokenizer([prompts[i] for i in batch_ids], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        outputs = model.generate(
            **inputs,
            do_sample=False,  # greedy decoding for reproducibility
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]  # same prompt length for the whole batch
        for i, text in zip(batch_ids, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
            answers[i] = text
        print(f"  generated {min(start + batch_size, len(order))}/{len(order)}")
    return answers
####


#### SCORING
def score_item(item: dict, text: str) -> dict:
    # one score dict per item, depending on its test set
    if item["test"].startswith("xlam"):
        return score_calls(text, item["reference"], convert_to_xlam_tool(item["tools"]))

    calls = parse_answer(text)
    if item["test"] == "bfcl":
        return score_bfcl(calls, item["reference"])

    # when2call: only "did it call or not" (an answer that doesn't parse counts as no call)
    called = calls is not None and len(calls) > 0
    return {"parse_ok": calls is not None, "called": called}


def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict]) -> dict:
    # plain averages per test set and category; confidence intervals come in step 2.4
    summary = {}
    for test in sorted({r["test"] for r in rows}):
        test_rows = [r for r in rows if r["test"] == test]
        if test.startswith("xlam"):
            summary[test] = {
                "n": len(test_rows),
                "parse_ok": mean(r["scores"]["parse_ok"] for r in test_rows),
                "exact_match": mean(r["scores"]["exact_match"] for r in test_rows),
                "name_match": mean(r["scores"]["name_match"] for r in test_rows),
                "invalid_call": mean(r["scores"]["invalid_call"] for r in test_rows),  # over parsed answers only
            }
        elif test == "bfcl":
            summary[test] = {}
            for category in sorted({r["category"] for r in test_rows}):
                cat_rows = [r for r in test_rows if r["category"] == category]
                summary[test][category] = {
                    "n": len(cat_rows),
                    "accuracy": mean(r["scores"]["correct"] for r in cat_rows),
                    "parse_ok": mean(r["scores"]["parse_ok"] for r in cat_rows),
                }
        elif test == "when2call":
            should_call = [r for r in test_rows if r["category"] == "tool_call"]
            should_not = [r for r in test_rows if r["category"] != "tool_call"]
            summary[test] = {
                "n": len(test_rows),
                "over_calling": mean(r["scores"]["called"] for r in should_not),
                "call_rate_on_tool_call_items": mean(r["scores"]["called"] for r in should_call),
                "parse_ok": mean(r["scores"]["parse_ok"] for r in test_rows),
            }
    return summary
####


#### MAIN
def evaluate(model_name: str, models_path: str, benchmarks_path: str, mode: str, limit: int | None,
             system_override: str | None, overwrite: bool):
    model_cfg = find_model(load_yaml(models_path), model_name)
    bench = load_yaml(benchmarks_path)
    cache_dir = REPO_ROOT / bench["xlam_cache"]

    system_message = read_system_message(cache_dir) if system_override is None else system_override

    # output folder: one per model and setting, never overwritten by accident
    suffix = ("_val" if mode == "val" else "") + ("_sysoverride" if system_override is not None else "") + (f"_limit{limit}" if limit else "")
    out_dir = REPO_ROOT / "eval" / f"{model_name}{suffix}"
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already exists: pass --overwrite to replace it")

    # the test sets: validation mode only looks at xlam's validation rows
    if mode == "val":
        sets = [load_xlam(cache_dir, "val")]
    else:
        sets = [load_xlam(cache_dir, "test"), load_bfcl(bench["bfcl"]), load_when2call(bench["when2call"])]
    items = [item for s in sets for item in (s[:limit] if limit else s)]
    print(f"{len(items)} items to evaluate")

    device = pick_device()
    tokenizer = get_tokenizer(model_cfg["path"], model_cfg["revision"])
    model = get_model(model_cfg["path"], model_cfg["revision"], device)

    gen = bench["generation"]
    prompts = render_prompts(items, tokenizer, system_message)
    texts = generate_answers(model, tokenizer, prompts, gen["batch_size"], gen["max_new_tokens"])

    rows = []
    for item, text in zip(items, texts):
        rows.append({
            "test": item["test"],
            "category": item["category"],
            "id": item["id"],
            "output": text,
            "calls": parse_answer(text),
            "scores": score_item(item, text),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "model": model_cfg,
        "mode": mode,
        "limit": limit,
        "system_message": system_message,
        "decoding": {"do_sample": False, **gen},
        "benchmarks": {"bfcl_commit": bench["bfcl"]["commit"], "when2call_commit": bench["when2call"]["commit"], "xlam_cache": bench["xlam_cache"]},
        "device": str(device),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": summarize(rows),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(json.dumps(metrics["results"], indent=2))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one model on the test sets (or on xlam validation)")
    parser.add_argument("--model", required=True, help="a name from configs/eval/models.yaml")
    parser.add_argument("--models", default="configs/eval/models.yaml")
    parser.add_argument("--benchmarks", default="configs/eval/benchmarks.yaml")
    parser.add_argument("--mode", choices=["test", "val"], default="test")
    parser.add_argument("--limit", type=int, default=None, help="only the first N items of each test set (smoke test)")
    parser.add_argument("--system-message", default=None, help="override the training system message (b0 side check)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    evaluate(args.model, args.models, args.benchmarks, args.mode, args.limit, args.system_message, args.overwrite)
####
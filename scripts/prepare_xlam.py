import yaml
from pathlib import Path
from datasets import load_dataset, Dataset
import hashlib
import json
from datetime import datetime, timezone
from tool_calling_post_training.tools import build_messages, format_answer
from tool_calling_post_training.load import get_tokenizer
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

def load_pinned_dataset(data_config_path: str):
    yaml_path = Path(data_config_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Could not find the data config file: {yaml_path}")
    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    seed = data_config["seed"]

    set_name = data_config["dataset"]["id"]
    revision = data_config["dataset"]["revision"]
    split = data_config["dataset"]["split"]
    cache_root = REPO_ROOT / data_config["cache"]["root"]
    hf_cache_dir = REPO_ROOT / data_config["cache"]["hf_cache"]

    sample_size = data_config["selection"]["sample_size"]
    val_size = data_config["selection"]["val_size"]
    test_size = data_config["selection"]["test_size"]

    tokenizer_id = data_config["tokenizer"]["id"]
    tokenizer_revision = data_config["tokenizer"]["revision"]

    tokenizer = get_tokenizer(tokenizer_id, tokenizer_revision)

    system_message = data_config["system_prompt"]
    max_seq_len = data_config["max_seq_len"]

    dataset = load_dataset(
        path=set_name,
        revision=revision,
        split=split,
        cache_dir=hf_cache_dir,
    )

    dataset = shuffle_and_pick(dataset, seed, sample_size)

    train, test, val = make_splits(dataset, test_size, val_size, seed)

    config_hash = compute_config_hash(data_config)
    data_dirname = f"{set_name.split('/')[-1]}_{config_hash}"
    split_dir = cache_root / data_dirname

    splits = {"train": train, "test": test, "val": val}

    stats = tokenize_and_save_splits(splits, split_dir, tokenizer, system_message, max_seq_len)

    write_dataset_manifest(split_dir, config_hash, data_config, splits, stats)

    return None

def shuffle_and_pick(data, seed, sample_size):
    if sample_size > len(data):
            raise ValueError(f"Asking for {sample_size} but dataset only contains {len(data)} samples.")
    # seeded shuffle
    dataset = data.shuffle(seed)

    # sample selection
    dataset = dataset.select(range(sample_size))
    return dataset

def make_splits(dataset: Dataset, test_size, val_size, seed):
    first = dataset.train_test_split(test_size=test_size, seed=seed)
    test = first["test"]
    second = first["train"].train_test_split(test_size = val_size, seed = seed)
    train = second["train"]
    val = second["test"]
    return train, test, val

def write_dataset_manifest(
    output_dir: Path,
    config_hash: str,
    data_config: dict,
    splits: dict[str, Dataset],
    stats: dict,
) -> Path:
    manifest = {
        "cache_dir": output_dir.name,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": data_config["seed"],
        "dataset": data_config["dataset"],
        "selection": data_config["selection"],
        "tokenizer": data_config["tokenizer"],
        "system_prompt": data_config["system_prompt"],
        "max_seq_len": data_config["max_seq_len"],
        "splits": {name: {"path": f"{name}/", **s} for name, s in stats.items()},
    }

    manifest_path = output_dir / "manifest.json"
    tmp_path = output_dir / "manifest.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    tmp_path.replace(manifest_path)

    return manifest_path

def compute_config_hash(data_config, length: int = 8) -> str:
    dump = json.dumps(data_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(dump).hexdigest()[:length]

def tokenize_row(row, tokenizer, system_message):
    messages = build_messages(row["tools"], row["query"], system_message)
    answer = format_answer(row["answers"])
    full_chat = messages + [{"role": "assistant", "content": answer}]

    prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    text_tokens = tokenizer.apply_chat_template(full_chat, tokenize=True)

    # the masking below assumes the prompt is the start of the full chat
    if text_tokens[:len(prompt_tokens)] != prompt_tokens:
        raise ValueError(f"Prompt is not a prefix of the full chat for query {row['query'][:80]!r}")

    labels = list(text_tokens)  # a copy
    labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)

    return {
        "input_ids": text_tokens,
        "labels": labels,
        "n_tokens": len(text_tokens),
        "n_learned": len(text_tokens) - len(prompt_tokens),
    }



def tokenize_and_save_splits(
        splits: dict,
        save_dir: Path,
        tokenizer: AutoTokenizer,
        system_message: str,
        max_seq_len: int = None
    ):

    stats = {}
    tokenized_splits = {}
    for name, split in splits.items():

        split.save_to_disk(save_dir / "raw" / name)   # the original query / tools / answers

        examples = []
        dropped = 0
        for row in split:
            tokenized = tokenize_row(row, tokenizer, system_message)
            if max_seq_len:
                if tokenized["n_tokens"]> max_seq_len:
                    dropped += 1
                    continue
            examples.append(tokenized)
        tokenized_splits[name] = Dataset.from_list(examples)
        tokenized_splits[name].save_to_disk(save_dir / name)
        n_tokens = sorted(e["n_tokens"] for e in examples)
        stats[name] = {
            "num_rows": len(tokenized_splits[name]),
            "columns": tokenized_splits[name].column_names,
            "dropped_too_long": dropped,
            "tokens_median": n_tokens[len(n_tokens) // 2],
            "tokens_p95": n_tokens[int(0.95 * len(n_tokens))],
            "learned_share": sum(e["n_learned"] for e in examples) / sum(n_tokens),
        }
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv)!=2:
        raise RuntimeError("Requires 1 argument as path to the data config yaml. Usage is:\n uv run scripts/prepare_xlam.py path/to/config.yaml")
    config_path = sys.argv[1]
    load_pinned_dataset(config_path)
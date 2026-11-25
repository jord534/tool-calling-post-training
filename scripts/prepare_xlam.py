import yaml
from pathlib import Path
from datasets import load_dataset, Dataset
import hashlib
import json
from datetime import datetime, timezone

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

    dataset = load_dataset(
        path=set_name,
        revision=revision,
        split=split,
        cache_dir=hf_cache_dir,
    )

    dataset = shuffle_and_pick(dataset, seed, sample_size)

    train, test, val = make_splits(dataset, test_size, val_size, seed)

    config_hash = compute_config_hash(data_config)
    data_dirname = f"{set_name.split("/")[-1]}_{config_hash}"
    split_dir = cache_root / data_dirname

    save_splits(split_dir, train = train, test = test, val = val)
    splits = {"train": train, "test": test, "val": val}
    write_dataset_manifest(split_dir, config_hash, data_config, splits)
    
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

def save_splits(save_dir, train: Dataset, test: Dataset, val: Dataset):
    for split_name, split_ds in (("train", train), ("test", test), ("val", val)):
        split_path = save_dir / split_name
        split_ds.save_to_disk(str(split_path))

def write_dataset_manifest(
    output_dir: Path,
    config_hash: str,
    data_config: dict,
    splits: dict[str, Dataset],
) -> Path:
    manifest = {
        "cache_dir": output_dir.name,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": data_config["seed"],
        "dataset": data_config["dataset"],
        "selection": data_config["selection"],
        "splits": {
            name: {
                "path": f"{name}/",
                "num_rows": len(ds),
                "columns": ds.column_names,
            }
            for name, ds in splits.items()
        },
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
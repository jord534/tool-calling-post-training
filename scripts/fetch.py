import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list, cwd: Path | None = None):
    # run a git command, stop with its message if it fails
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def fetch(repo: str, commit: str, target: Path, sparse_path: str | None = None):
    # clone repo at a fixed commit; with sparse_path, only that folder is downloaded
    if target.exists():
        head = run(["git", "rev-parse", "HEAD"], cwd=target)
        if head != commit:
            raise RuntimeError(f"{target} is at {head}, expected {commit}: delete the folder and run again")
        print(f"already there: {target} at {commit[:8]}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if sparse_path:
        run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(target)])
        run(["git", "sparse-checkout", "set", sparse_path], cwd=target)
    else:
        run(["git", "clone", "--no-checkout", repo, str(target)])
    run(["git", "checkout", "--quiet", commit], cwd=target)
    print(f"fetched: {target} at {commit[:8]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("Usage:\n  uv run python scripts/fetch_benchmarks.py configs/eval/benchmarks.yaml")
    with open(sys.argv[1]) as f:
        bench = yaml.safe_load(f)

    fetch(bench["bfcl"]["repo"], bench["bfcl"]["commit"], REPO_ROOT / bench["bfcl"]["dir"], bench["bfcl"]["sparse_path"])
    fetch(bench["when2call"]["repo"], bench["when2call"]["commit"], REPO_ROOT / bench["when2call"]["dir"])
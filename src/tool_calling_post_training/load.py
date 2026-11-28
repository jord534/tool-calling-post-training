import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tokenizer(id, revision):
    return AutoTokenizer.from_pretrained(id, revision = revision)


def pick_device() -> torch.device:
    # the best device available: cuda on the rented gpus, mps on the mac, else cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(id, revision, device: torch.device):
    # id: a hugging face id (b0, ref) or a local export folder (sft), revision: a commit hash or None
    # bf16 on cuda for speed; fp32 elsewhere (the m1 has no bf16)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(id, revision = revision, dtype = dtype)
    return model.to(device).eval()
from transformers import AutoTokenizer

def get_tokenizer(id, revision):
    return AutoTokenizer.from_pretrained(id, revision = revision)


def get_model():
    return None
    
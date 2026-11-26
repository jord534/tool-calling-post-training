import json
from collections import defaultdict

def parse_answer(text: str):
    # returns the list of calls, [] for "no tool fits", or None if the answer isn't valid
    # a bad answer is a result to count, not an error: never raise on model output
    text = text.strip()
    try:
        js = json.loads(text)
    except json.JSONDecodeError:
        # print(f"Could not parse the answer. There is something wrong with it: {text!r}")
        return None

    if not isinstance(js, dict) or "tool_calls" not in js:
        # print(f"Parsing the text did not yiled a dict, got a {type(js).__name__}")
        #or
        # print("Did not find a tool_calls key in the dict")
        return None
    calls = js["tool_calls"]
    if not isinstance(calls, list):
        # print(f"tool_calls is not a list but a {type(js["tool_calls"]).__name__}")
        
        return None
    for call in calls:
        if not isinstance(call, dict):
            # print(f"Every item is the tool_calls dict must be a dict, found a {type(tool).__name__}")
            return None
        if not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
            # print(f"Missing name or argument in the tool {tool}.")
            return None
    return calls



def standardize(call: dict) -> str:
    # one call as text with sorted keys: equal calls give equal strings
    return json.dumps({"name": call["name"], "arguments": call["arguments"]}, sort_keys=True)


def score_calls(predicted: str, reference: list, tools: list) -> dict:
    # predicted: the raw text the model wrote
    # reference: the correct calls, parsed (list of {"name", "arguments"})
    # tools: the tools offered, parsed, in xlam format (name -> parameters dict)
    calls = parse_answer(predicted)
    if calls is None:
        # not a valid answer: nothing else can be judged
        return {"parse_ok": False, "invalid_call": None, "name_match": False, "exact_match": False}

    # invalid = a tool that wasn't offered, or an argument the tool doesn't have
    params_per_tool = {tool["name"]: set(tool["parameters"].keys()) for tool in tools}
    invalid_call = False
    for call in calls:
        if call["name"] not in params_per_tool:
            invalid_call = True
        elif not set(call["arguments"]) <= params_per_tool[call["name"]]:
            invalid_call = True

    # order of calls doesn't matter, how many times each one appears does
    name_match = sorted(c["name"] for c in calls) == sorted(c["name"] for c in reference)
    exact_match = sorted(standardize(c) for c in calls) == sorted(standardize(c) for c in reference)

    return {
        "parse_ok": True,
        "invalid_call": invalid_call,
        "name_match": name_match,
        "exact_match": exact_match,
    }
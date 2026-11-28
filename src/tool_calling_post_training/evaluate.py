import json
import re
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




#### BFCL SCORING
# a simplified port of BFCL's python ast checker (bfcl_eval/eval_checker/ast_eval/ast_checker.py)
# ground truth format, one dict per expected call: [{"func_name": {"param": [accepted values]}}]
# "" among the accepted values means the parameter is optional: it may be left out
# for irrelevance items there is no ground truth: pass None


def _standardize(text: str) -> str:
    # same as bfcl: ignore spaces, ",./-_*^", case, and quote style
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", text).lower().replace("'", '"')


def _same_kind(value, accepted) -> bool:
    # bool is a subclass of int in python: keep them apart, so True never matches 1
    if isinstance(value, bool) or isinstance(accepted, bool):
        return isinstance(value, bool) and isinstance(accepted, bool)
    # an int is fine where a float is expected (bfcl allows it)
    if isinstance(value, (int, float)) and isinstance(accepted, (int, float)):
        return not (isinstance(value, float) and isinstance(accepted, int) and not isinstance(accepted, bool))
    return type(value) == type(accepted)


def _value_ok(value, accepted_values: list) -> bool:
    # true if the value matches one of the accepted values
    for accepted in accepted_values:
        if accepted == "" and value != "":
            continue  # "" only marks the parameter as optional
        if isinstance(value, str) and isinstance(accepted, str):
            if _standardize(value) == _standardize(accepted):
                return True
        elif isinstance(value, list) and isinstance(accepted, list):
            if len(value) == len(accepted) and all(
                _value_ok(v, [a]) for v, a in zip(value, accepted)
            ):
                return True
        elif isinstance(value, dict) and isinstance(accepted, dict):
            # each key of a dict answer holds its own list of accepted values
            if _arguments_ok(value, accepted):
                return True
        elif _same_kind(value, accepted) and value == accepted:
            return True
    return False


def _arguments_ok(arguments: dict, expected: dict) -> bool:
    # every given argument is expected and has an accepted value
    for param, value in arguments.items():
        if param not in expected or not _value_ok(value, expected[param]):
            return False
    # every expected argument left out must be optional
    for param, accepted_values in expected.items():
        if param not in arguments and "" not in accepted_values:
            return False
    return True


def _call_ok(call: dict, expected_call: dict) -> bool:
    # expected_call has one key: the function name
    (func_name, expected_args), = expected_call.items()
    return call["name"] == func_name and _arguments_ok(call["arguments"], expected_args)


def score_bfcl(calls, ground_truth) -> dict:
    # calls: the output of parse_answer (None if the answer didn't parse)
    # ground_truth: the item's "ground_truth" list from bfcl's possible_answer file, or None for irrelevance
    parse_ok = calls is not None

    # irrelevance: correct if the model made no call (bfcl also counts an unparsable answer as no call)
    if ground_truth is None:
        correct = (not parse_ok) or len(calls) == 0
        return {"parse_ok": parse_ok, "correct": correct, "error": None if correct else "called_when_irrelevant"}

    if not parse_ok:
        return {"parse_ok": False, "correct": False, "error": "parse_failed"}
    if len(calls) != len(ground_truth):
        return {"parse_ok": True, "correct": False, "error": "wrong_count"}

    # each expected call must match a different model call, in any order
    unmatched = list(range(len(calls)))
    for expected_call in ground_truth:
        match = next((i for i in unmatched if _call_ok(calls[i], expected_call)), None)
        if match is None:
            func_name = list(expected_call.keys())[0]
            return {"parse_ok": True, "correct": False, "error": f"no_match:{func_name}"}
        unmatched.remove(match)

    return {"parse_ok": True, "correct": True, "error": None}
####
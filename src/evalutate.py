import json

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
import json


#### XLAM INSTRUCTIONS
# paste both texts word for word from https://huggingface.co/Salesforce/xLAM-1b-fc-r (usage example)
# same text for training, the untouched model, your model and xlam-1b: any change makes the comparison unfair
TASK_INSTRUCTION = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.
""".strip()

FORMAT_INSTRUCTION = """
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list '[]'.
```
{
    "tool_calls": [
    {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
    ... (more tool calls as required)
    ]
}
```
""".strip()


#### PARSING
def parse_json_field(value):
    # xlam stores tools and answers as json strings: turn them into python lists
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Expected a JSON string, could not parse it ({e}): {value[:200]}")
    if isinstance(value, (list, dict)):
        return value
    raise TypeError(f"Expected a JSON string, list or dict, got {type(value).__name__}")
####


#### TOOLS
def convert_to_xlam_tool(tools):
    # adapted from https://huggingface.co/Salesforce/xLAM-1b-fc-r
    # openai / bfcl style tools have their arguments under parameters.properties: flatten them
    # xlam training tools are already flat (arguments directly under parameters): keep them as they are
    # like xlam's own code, the flattening drops "required": all three models then see the same tools
    if isinstance(tools, list):
        return [convert_to_xlam_tool(tool) for tool in tools]
    if not isinstance(tools, dict):
        raise TypeError(f"Expected a tool (dict) or a list of tools, got {type(tools).__name__}: parse json strings first")

    tool = tools.get("function", tools)  # some files wrap the tool in {"type": "function", "function": {...}}
    if "name" not in tool:
        raise ValueError(f"Expected a tool with a 'name', got keys {list(tool.keys())}")

    params = tool.get("parameters", {})
    if "properties" not in params:
        return tool  # already xlam format

    return { # open AI or BFCL format case
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": {k: v for k, v in params["properties"].items()},
    }
####


#### PROMPT
def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str) -> str:
    # copied from https://huggingface.co/Salesforce/xLAM-1b-fc-r
    xlam_format_tools = convert_to_xlam_tool(tools)
    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(xlam_format_tools)}\n[END OF AVAILABLE TOOLS]\n\n"
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt


def build_messages(tools: list, query: str, system_message: str | None = None) -> list[dict]:
    # the chat to feed apply_chat_template: the whole prompt goes in one user message, as on the xlam card
    # system_message=None lets the template insert its default system prompt (deepseek: "computer science only")
    # a custom system message must end with "\n" then the deepseek template glues it to "### Instruction:"
    if not isinstance(query, str) or not query:
        raise ValueError(f"Expected a non-empty query str, got {query!r}")

    content = build_prompt(TASK_INSTRUCTION, FORMAT_INSTRUCTION, parse_json_field(tools), query)
    messages = [{"role": "user", "content": content}]
    if system_message is not None:
        if not system_message.endswith("\n"):
            raise ValueError(f"The system message should end with a newline, got {system_message[-20:]!r}")
        messages = [{"role": "system", "content": system_message}] + messages
    return messages
####


#### ANSWER
def format_answer(calls) -> str:
    # the exact text the model must learn to write; an empty list means "no tool fits"
    calls = parse_json_field(calls)
    if not isinstance(calls, list):
        raise TypeError(f"Expected a list of calls, got {type(calls).__name__}")
    clean = []
    for call in calls:
        if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
            raise ValueError(f"Expected each call as {{'name': ..., 'arguments': ...}}, got {call!r}")
        clean.append({"name": call["name"], "arguments": call["arguments"]})  # name first, always
    return json.dumps({"tool_calls": clean})

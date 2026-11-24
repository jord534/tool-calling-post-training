import json

# Helper function to convert openai format tools to our more concise xLAM format
def convert_to_xlam_tool(tools):
    # copied from https://huggingface.co/Salesforce/xLAM-1b-fc-r
    if isinstance(tools, dict):
        params = tools.get("parameters", {})
        if "properties" not in params:  # already xlam format (training data)
            return tools
        return {
            "name": tools["name"],
            "description": tools["description"],
            "parameters": {k: v for k, v in tools["parameters"].get("properties", {}).items()}
        }
    elif isinstance(tools, list):
        return [convert_to_xlam_tool(tool) for tool in tools]
    else:
        return tools

# Helper function to build the input prompt for our model
def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str):
    # copied from https://huggingface.co/Salesforce/xLAM-1b-fc-r
    xlam_format_tools = convert_to_xlam_tool(tools)
    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(xlam_format_tools)}\n[END OF AVAILABLE TOOLS]\n\n"
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt

def format_answer(calls: list):
    return {"tool_calls": calls}
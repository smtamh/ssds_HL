## convert tools format from mcp to vllm (openai)
def clean_schema(schema: dict):
    schema = dict(schema)
    schema.pop("title", None)
    return schema

def tool_conversion_mcp_vllm(mcp_tools):
    out = []
    for t in mcp_tools:
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "").strip(),
                "parameters": clean_schema(t.inputSchema) or {"type": "object", "properties": {}, "required": []},
            },
        })
    return out
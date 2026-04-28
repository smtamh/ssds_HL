import asyncio
from client import MCPClient
import argparse
import json
import time

import config
from utils import tool_conversion_mcp_vllm

class InferenceNode:
    def __init__(self, mcp_server: str = "server.py", transport: str = "stdio"):
        from vllm import LLM, SamplingParams
        from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

        self.sampling_params = SamplingParams(**config.SAMPLING_PARAMS)
        self.llm = LLM(
            model=config.LLM_PATH,
            generation_config="vllm",
            gpu_memory_utilization=config.GPU_UTIL,
            max_model_len=config.MAX_LENGTH,
            max_num_seqs=config.MAX_NUM_SEQS
        )
        self.parser = Hermes2ProToolParser(tokenizer=self.llm.get_tokenizer())
        self.mcp = MCPClient(mcp_server, transport=transport)

    @staticmethod
    async def handle_tool_calls(mcp_client: MCPClient, tool_calls: list[dict]):
        results = []
        for tc in tool_calls:
            name = tc.function.name
            args  = tc.function.arguments

            if isinstance(args, str):
                args=json.loads(args)
            result = await mcp_client.call_tool(name, args)   
            results.append({
                "id": tc.id,
                "function": name,
                "arguments": args,
                "result": result.structuredContent['result']
            })

        return results

    async def arun(self):
        await self.mcp.connect()
        tools = await self.mcp.list_tools()
        tools = tool_conversion_mcp_vllm(tools.tools)

        print("✅ Connected to MCP Server")
        print("=" * 50)
        print("Interactive Inference Mode")
        print("Enter 'quit' or 'exit' to stop")
        print("=" * 50)
        
        try:
            while True:
                user_input = input("\n👤 Enter your query: ")

                # terminate condition
                if user_input.strip().lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break
                
                if not user_input.strip():
                    continue
                
                messages = [
                    {
                        "role": "user",
                        "content": user_input.strip(),
                    }
                ]

                print("\n🤖 Starting Inference...")
                t1 = time.time()
                outputs = self.llm.chat(messages, sampling_params=self.sampling_params, tools=tools)
                t2 = time.time()

                output = outputs[0].outputs[0].text.strip()

                print(f"\n📝 Response:\n{output}")
                print(f"\n⏱️ Inference Time: {t2 - t1:.2f} seconds")

                # Tool check and execution
                """
                tools_called: True / False
                tool_calls:   [ToolCall(id, type='function', function=FunctionCall(name, arguments)]
                content:      NL response (including <think>)
                """
                info = self.parser.extract_tool_calls(output, request=None)

                if info.tools_called:
                    tool_results = await self.handle_tool_calls(self.mcp, info.tool_calls)
                    print("\n🛠️ Tool Execution Results:\n", tool_results)
                else:
                    pass

        finally:
            await self.mcp.cleanup()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport type. Use stdio for local server.py, or streamable-http for an already running MCP server.",
    )
    parser.add_argument(
        "--mcp-server",
        default=None,
        help="MCP server target. Defaults to server.py for stdio and http://127.0.0.1:8000/mcp for streamable-http.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()
        mcp_server = args.mcp_server
        if mcp_server is None:
            if args.transport == "streamable-http":
                # Check the MCP endpoint exposed by server.py in the container,
                # and update mcp_server here if the host/port/path is different.
                mcp_server = "http://127.0.0.1:8000/mcp"
            else:
                mcp_server = "server.py"

        print("🚀 Initializing Inference Node...")
        inference_node = InferenceNode(
            mcp_server=mcp_server,
            transport=args.transport,
        )
        asyncio.run(inference_node.arun())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

import asyncio
from client import MCPClient
import argparse
import json
import socket
import sys
import time

import config
from utils import tool_conversion_mcp_vllm

class InferenceNode:
    def __init__(
        self,
        mcp_server: str = "http://127.0.0.1:8000/mcp",
        transport: str = "streamable-http",
        input_source: str = "text",
        speech_host: str = "127.0.0.1",
        speech_port: int = 8765,
    ):
        from vllm import LLM, SamplingParams
        from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

        self.input_source = input_source
        self.speech_host = speech_host
        self.speech_port = speech_port
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
            print("==================")
            print(result)
            print("==================")
            results.append({
                "id": tc.id,
                "function": name,
                "arguments": args,
                "result": result.structuredContent['result']
            })

        return results

    def read_user_input(self, speech_stream=None) -> str | None:
        if self.input_source == "text":
            return input("\n👤 Enter your query: ")

        if self.input_source == "stdin":
            line = sys.stdin.readline()
            if not line:
                return None
            user_input = line.strip()
            print(f"\n👤 Recognized input: {user_input}", flush=True)
            return user_input

        if self.input_source == "speech":
            if speech_stream is None:
                raise RuntimeError("Speech input stream is not connected")
            print("\n🎙️ Waiting for speech...", flush=True)
            line = speech_stream.readline()
            if not line:
                return None
            user_input = line.strip()
            print(f"\n👤 Recognized input: {user_input}", flush=True)
            return user_input

        raise ValueError(f"Unsupported input source: {self.input_source}")

    def connect_speech_stream(self):
        if self.input_source != "speech":
            return None, None

        speech_socket = socket.create_connection((self.speech_host, self.speech_port))
        speech_stream = speech_socket.makefile("r", encoding="utf-8")
        print(
            f"🎙️ Connected to speech recognizer at {self.speech_host}:{self.speech_port}",
            flush=True,
        )
        return speech_socket, speech_stream

    async def arun(self):
        await self.mcp.connect()
        tools = await self.mcp.list_tools()
        tools = tool_conversion_mcp_vllm(tools.tools)
        
        print("========================================================")
        for ttt in tools:
            print(ttt, "\n\n")
        print("========================================================")
        
        speech_socket, speech_stream = self.connect_speech_stream()

        print("✅ Connected to MCP Server")
        print("=" * 50)
        print("Interactive Inference Mode")
        if self.input_source == "text":
            print("Enter 'quit' or 'exit' to stop")
        else:
            print("Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            while True:
                user_input = self.read_user_input(speech_stream)
                if user_input is None:
                    print("Input stream closed. Exiting...")
                    break

                # terminate condition
                if user_input.strip().lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    # 루프를 완전히 종료하기 위해 커스텀 예외 활용 또는 sys.exit() 처리 가능
                    sys.exit(0)
                
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

                info = self.parser.extract_tool_calls(output, request=None)

                if info.tools_called:
                    tool_results = await self.handle_tool_calls(self.mcp, info.tool_calls)
                    print("\n🛠️ Tool Execution Results:\n", tool_results)
                else:
                    pass

        finally:
            if speech_stream is not None:
                speech_stream.close()
            if speech_socket is not None:
                speech_socket.close()
            await self.mcp.cleanup()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="MCP transport type (default: streamable-http). Use stdio to launch server.py as a subprocess.",
    )
    parser.add_argument(
        "--mcp-server",
        default=None,
        help="MCP server target. Defaults to server.py for stdio and http://127.0.0.1:8000/mcp for streamable-http.",
    )
    parser.add_argument(
        "--input-source",
        choices=["text", "speech", "stdin"],
        default="text",
        help=(
            "Human input source. Use text for keyboard input, speech to connect "
            "to recognize_speech.py --server, or stdin for piped recognized text."
        ),
    )
    parser.add_argument("--speech-host", default="127.0.0.1")
    parser.add_argument("--speech-port", type=int, default=8765)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mcp_server = args.mcp_server
    if mcp_server is None:
        if args.transport == "streamable-http":
            mcp_server = "http://127.0.0.1:8000/mcp"
        else:
            mcp_server = "server.py"

    # 에러 발생 시 무한 재시작을 위한 루프
    # while True:
    try:
        print("\n🚀 Initializing Inference Node...")
        inference_node = InferenceNode(
            mcp_server=mcp_server,
            transport=args.transport,
            input_source=args.input_source,
            speech_host=args.speech_host,
            speech_port=args.speech_port,
        )
        asyncio.run(inference_node.arun())
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting permanently...")
    #     break
        
    # except SystemExit:
    #     # 사용자가 quit/exit을 입력하여 종료된 경우
    #     break
        
    # except Exception as e:
    #     print(f"\n❌ Error occurred: {e}")
    #     import traceback
    #     traceback.print_exc()
        
    #     # 연속적인 에러 폭발로 인한 시스템 과부하 방지를 위해 5초 대기 후 재시작
    #     print("\n🔄 Restarting engine in 5 seconds...")
    #     time.sleep(5)

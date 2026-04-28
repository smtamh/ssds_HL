import os
from contextlib import AsyncExitStack
from typing import Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(
        self,
        server: str,
        transport: Literal["stdio", "streamable-http"] = "stdio",
    ):
        self.server = server
        self.transport = transport
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect(self):
        if self.transport == "stdio":
            await self._connect_stdio()
        elif self.transport == "streamable-http":
            await self._connect_streamable_http()
        else:
            raise ValueError(
                f"Unsupported MCP transport '{self.transport}'. "
                "Use 'stdio' or 'streamable-http'."
            )

        await self.session.initialize()

    async def _connect_stdio(self):
        server_params = StdioServerParameters(
            command="uv",
            args=["run", self.server],
            env=os.environ.copy(),
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session= await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

    async def _connect_streamable_http(self):
        http_transport = await self.exit_stack.enter_async_context(
            streamable_http_client(self.server)
        )
        self.http_read, self.http_write, self.get_session_id = http_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.http_read, self.http_write)
        )

    async def list_tools(self):
        assert self.session is not None
        return await self.session.list_tools()

    async def call_tool(self, name: str, arguments: dict):
        assert self.session is not None
        return await self.session.call_tool(name, arguments)
    
    async def cleanup(self):
        await self.exit_stack.aclose()

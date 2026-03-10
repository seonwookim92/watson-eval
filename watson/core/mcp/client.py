import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from core.config import config

class MCPClient:
    def __init__(self):
        mcp_dir = os.path.dirname(config.MCP_SERVER_PATH)
        self.server_params = StdioServerParameters(
            command="/opt/anaconda3/envs/cyber-ontology-env/bin/python",
            args=[config.MCP_SERVER_PATH],
            env={**os.environ, "PYTHONPATH": mcp_dir, "ONTOLOGY_DIR": config.ONTOLOGY_DIR},
        )

        self.session = None
        self._stdio_mgr = None

    async def connect(self):
        if self.session:
            return
        self._stdio_mgr = stdio_client(self.server_params)
        read, write = await self._stdio_mgr.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()

    async def disconnect(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
            await self._stdio_mgr.__aexit__(None, None, None)
            self.session = None

    async def list_tools(self):
        if not self.session:
            await self.connect()
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, arguments: dict):
        if not self.session:
            await self.connect()
        result = await self.session.call_tool(tool_name, arguments)
        return result.content

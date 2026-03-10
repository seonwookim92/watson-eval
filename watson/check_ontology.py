import asyncio
from core.mcp.client import MCPClient

async def check():
    client = MCPClient()
    await client.connect()
    try:
        print("\n--- Searching for 'Malware' ---")
        res = await client.call_tool("search_classes", {"query": "Malware"})
        print(res)
        
        print("\n--- Searching for 'IP' ---")
        res = await client.call_tool("search_classes", {"query": "IP"})
        print(res)

        print("\n--- Searching for 'File' ---")
        res = await client.call_tool("search_classes", {"query": "File"})
        print(res)
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(check())

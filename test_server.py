import subprocess
import json
import sys
import time

def test_mcp_server():
    # Start the server process
    process = subprocess.Popen(
        [sys.executable, 'server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1
    )

    print("Server started. Sending initialize request...")

    # 1. Initialize
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    }
    
    process.stdin.write(json.dumps(init_request) + "\n")
    process.stdin.flush()

    # Read response (might get some logs first, so we need to be careful)
    # In a real scenario we'd use a proper JSON-RPC client, but this is a quick check
    
    # We'll just read a line and hope it's the response
    response_line = process.stdout.readline()
    print(f"Response: {response_line}")
    
    # 2. List Tools
    list_tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    process.stdin.write(json.dumps(list_tools_request) + "\n")
    process.stdin.flush()
    
    response_line = process.stdout.readline()
    print(f"Tools List Response: {response_line}")

    # 3. Call get_account_info
    call_tool_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_account_info",
            "arguments": {}
        }
    }
    process.stdin.write(json.dumps(call_tool_request) + "\n")
    process.stdin.flush()
    
    # This might take a moment if it needs to connect
    response_line = process.stdout.readline()
    print(f"Call Tool Response (Account): {response_line}")

    # 4. Call get_stock_price
    call_tool_request_2 = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "get_stock_price",
            "arguments": {"symbol": "BHP"}
        }
    }
    process.stdin.write(json.dumps(call_tool_request_2) + "\n")
    process.stdin.flush()
    
    response_line = process.stdout.readline()
    print(f"Call Tool Response (Stock): {response_line}")

    process.terminate()

if __name__ == "__main__":
    test_mcp_server()

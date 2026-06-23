"""
Run the yunwen knowledge-base MCP service.

Usage:
    cd yunwen
    python run_mcp_server.py

Service URL:
    http://127.0.0.1:9100/mcp
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

yunwen_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(yunwen_dir))
os.chdir(yunwen_dir)
load_dotenv(yunwen_dir / ".env")

from app.mcp_server import mcp


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=9100,
        path="/mcp",
    )

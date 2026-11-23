# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
import asyncio

import pytest

pytest.importorskip("mcp")


# REGRESSION (0.4): imported mcp.server.fastmcp, which mcp 2.x removed, so the MCP server
# crashed on startup with a current `pip install mcp`.
def test_server_imports_and_registers_tools():
    from chatterbox_studio import mcp_server as m
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert {"status", "list_voices", "speak", "speak_long", "clone_voice"} <= names
    assert "delete_voice" not in names          # deny-first: never exposed to the model


def test_friendly_error_when_app_not_running(monkeypatch):
    from chatterbox_studio import mcp_server as m
    monkeypatch.setattr(m, "BASE", "http://127.0.0.1:9")
    with pytest.raises(RuntimeError, match="isn't running"):
        m.status()

"""MCP server that drives a running Houdini session.

The Houdini half is houdini_bridge.py, which listens on 127.0.0.1:9877 inside
Houdini. This process speaks MCP on stdin/stdout and forwards each call to it.

Run it with the project venv (it has the mcp package):
    <project>\\tools\\venv\\Scripts\\python.exe mcp_server.py
"""
import json
import os
import socket

from mcp.server.mcpserver import MCPServer

HOST = "127.0.0.1"
# Must match houdini_bridge.PORT. 9876 belongs to Blender's MCP add-on.
PORT = int(os.environ.get("HOUDINI_BRIDGE_PORT", 9877))
TIMEOUT = 600.0  # a render or a training step can keep Houdini busy for a while

server = MCPServer(
    name="houdini",
    instructions=(
        "Drives the Houdini session open on this machine. Houdini must be running "
        "with the GSplat Studio package loaded, which starts the bridge at launch. "
        "If a call reports the bridge is unreachable, ask the user to start Houdini "
        "rather than guessing at the scene."
    ),
)


def call(command: str, **params) -> dict:
    """Send one command to Houdini and return its result."""
    request = json.dumps({"type": command, "params": params}) + "\n"
    try:
        with socket.create_connection((HOST, PORT), timeout=10.0) as sock:
            sock.settimeout(TIMEOUT)
            sock.sendall(request.encode("utf-8"))
            with sock.makefile("rb") as stream:
                line = stream.readline()
    except OSError as err:
        raise RuntimeError(
            f"cannot reach Houdini on {HOST}:{PORT} ({err}). Houdini is probably "
            "closed, or it started before the bridge was installed."
        ) from err

    if not line:
        raise RuntimeError("Houdini closed the connection without answering")
    reply = json.loads(line.decode("utf-8"))
    if reply.get("status") != "success":
        raise RuntimeError(reply.get("message", "unknown error from Houdini"))
    return reply.get("result", {})


@server.tool(description="Check that Houdini is reachable; returns version, licence and open file.")
def houdini_status() -> dict:
    return call("ping")


@server.tool(description="List the nodes in /obj, /stage, /out and /mat, plus the current selection.")
def houdini_scene_info() -> dict:
    return call("scene_info")


@server.tool(description="List the Python Panels this Houdini knows about.")
def houdini_list_panels() -> dict:
    return call("list_panels")


@server.tool(description="Open a Python Panel as a floating window, e.g. 'gsplat_studio'.")
def houdini_open_panel(name: str = "gsplat_studio", width: int = 560, height: int = 820) -> dict:
    return call("open_panel", name=name, width=width, height=height)


@server.tool(description=(
    "Run Python inside Houdini. `hou` is already imported. Assign to `result` to get "
    "a value back; printed text comes back as `output`."
))
def houdini_run_python(code: str) -> dict:
    return call("run_python", code=code)


@server.tool(description="Save what the 3D viewport is showing to a PNG and return its path.")
def houdini_viewport_screenshot(path: str = "") -> dict:
    return call("screenshot", path=path)


if __name__ == "__main__":
    server.run("stdio")

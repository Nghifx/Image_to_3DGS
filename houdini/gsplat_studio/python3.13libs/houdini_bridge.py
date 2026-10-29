"""A local command bridge so an outside tool can drive this Houdini session.

Houdini has no MCP server of its own, so this is the Houdini half: a small
socket server that listens on localhost only and runs each request on Houdini's
main thread. The MCP half (mcp_server.py) talks to it and speaks MCP to Claude.

Start it from Houdini (scripts/123.py does this at launch):

    import houdini_bridge
    houdini_bridge.start()

Messages are newline-delimited JSON, one request per line:
    {"type": "ping", "params": {}}
    -> {"status": "success", "result": {...}}

Anything on this port can run Python inside Houdini, so it binds 127.0.0.1 and
nothing else; it is for tools on this machine.
"""
import json
import os
import socket
import threading
import traceback

import hou

HOST = "127.0.0.1"
# 9876 is Blender's MCP add-on port. Windows lets a second process bind a port
# that is already taken, so sharing it silently sent our commands to Blender;
# keep this one to ourselves. Override with HOUDINI_BRIDGE_PORT if it clashes.
PORT = int(os.environ.get("HOUDINI_BRIDGE_PORT", 9877))
ROOT = os.environ.get("GSPLAT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOG = os.path.join(ROOT, "logs", "houdini_bridge.log")

_server = None


def log(message: str) -> None:
    import datetime

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{stamp} {message.rstrip()}\n")
    print("[houdini_bridge]", message, flush=True)


def _on_main_thread(func):
    """Run func on Houdini's main thread and return its value.

    Touching the scene from a socket thread corrupts Houdini, so every command
    is handed to the UI thread and waited on.
    """
    import hdefereval

    return hdefereval.executeInMainThreadWithResult(func)


# -- commands --------------------------------------------------------------


def cmd_ping(_params: dict) -> dict:
    return {
        "houdini": hou.applicationVersionString(),
        "license": str(hou.licenseCategory()),
        "hip": hou.hipFile.name(),
        "frame": hou.frame(),
    }


def cmd_scene_info(_params: dict) -> dict:
    def collect():
        info = {"hip": hou.hipFile.name(), "frame": hou.frame(), "contexts": {}}
        for path in ("/obj", "/stage", "/out", "/mat"):
            parent = hou.node(path)
            if parent is None:
                continue
            info["contexts"][path] = [
                {"name": n.name(), "type": n.type().name()} for n in parent.children()
            ]
        info["selected"] = [n.path() for n in hou.selectedNodes()]
        return info

    return _on_main_thread(collect)


def cmd_run_python(params: dict) -> dict:
    """Run Python inside Houdini. Whatever the code puts in `result` comes back."""
    code = params.get("code", "")

    def run():
        import contextlib
        import io

        namespace = {"hou": hou, "result": None}
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(code, namespace)
        value = namespace.get("result")
        try:
            json.dumps(value)
        except TypeError:
            value = repr(value)
        return {"output": out.getvalue(), "result": value}

    return _on_main_thread(run)


def cmd_open_panel(params: dict) -> dict:
    """Open a Python Panel as a floating window, e.g. the GSplat Studio panel."""
    name = params.get("name", "gsplat_studio")
    width = int(params.get("width", 560))
    height = int(params.get("height", 820))

    def open_it():
        interface = hou.pypanel.interfaceByName(name)
        if interface is None:
            raise hou.Error(f"no Python Panel named {name}; "
                            f"available: {sorted(hou.pypanel.interfaces())}")
        tab = hou.ui.curDesktop().createFloatingPaneTab(
            hou.paneTabType.PythonPanel, size=(width, height))
        tab.setActiveInterface(interface)
        return {"opened": name, "pane": tab.name()}

    return _on_main_thread(open_it)


def cmd_list_panels(_params: dict) -> dict:
    return {"panels": sorted(hou.pypanel.interfaces())}


def cmd_screenshot(params: dict) -> dict:
    """Save what the 3D viewport is showing, so the caller can look at it."""
    path = params.get("path") or os.path.join(
        ROOT, "logs", "viewport.png")

    def shoot():
        viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
        if viewer is None:
            raise hou.Error("no scene viewer open")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        settings = viewer.flipbookSettings().stash()
        settings.output(path.replace("\\", "/"))
        settings.frameRange((hou.frame(), hou.frame()))
        settings.outputToMPlay(False)
        viewer.flipbook(viewer.curViewport(), settings)
        return {"path": path}

    return _on_main_thread(shoot)


COMMANDS = {
    "ping": cmd_ping,
    "scene_info": cmd_scene_info,
    "run_python": cmd_run_python,
    "open_panel": cmd_open_panel,
    "list_panels": cmd_list_panels,
    "screenshot": cmd_screenshot,
}


# -- server ----------------------------------------------------------------


class Bridge:
    def __init__(self, host: str = HOST, port: int = PORT):
        self.host = host
        self.port = port
        self.socket: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.running = False

    def start(self) -> None:
        if self.running:
            log(f"already listening on {self.host}:{self.port}")
            return
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name == "nt":
            # Windows otherwise lets us bind a port another program already has,
            # and then quietly delivers the connections to it. Fail loudly instead.
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        log(f"listening on {self.host}:{self.port}")

    def stop(self) -> None:
        self.running = False
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        log("stopped")

    def _accept_loop(self) -> None:
        while self.running:
            try:
                client, _ = self.socket.accept()
            except OSError:
                break
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        with client, client.makefile("rwb") as stream:
            for raw in stream:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                stream.write((json.dumps(self._handle(line)) + "\n").encode("utf-8"))
                stream.flush()

    def _handle(self, line: str) -> dict:
        try:
            request = json.loads(line)
            handler = COMMANDS.get(request.get("type"))
            if handler is None:
                return {"status": "error",
                        "message": f"unknown command; have {sorted(COMMANDS)}"}
            return {"status": "success", "result": handler(request.get("params") or {})}
        except Exception as err:
            log("error: " + traceback.format_exc())
            return {"status": "error", "message": f"{type(err).__name__}: {err}"}


def start(port: int = PORT) -> None:
    global _server
    if _server is None:
        _server = Bridge(port=port)
    _server.start()


def stop() -> None:
    if _server is not None:
        _server.stop()

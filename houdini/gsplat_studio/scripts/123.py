"""Houdini startup: bring up the local command bridge so tools can drive this session."""
import traceback

try:
    import houdini_bridge
    houdini_bridge.start()
except Exception:
    print("[gsplat_studio] bridge failed to start:\n" + traceback.format_exc())

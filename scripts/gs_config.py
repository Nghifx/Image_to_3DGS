"""Shared settings for the Wukong Gaussian-splat pipeline (Houdini 21 + Brush)."""
import os

SOURCE_HIP = r"N:\01_3D Model\01_Animation\06_Character_animation\Wukong_agent_ldev.hip"

ROOT = os.environ.get("GSPLAT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS_HIP = os.path.join(ROOT, "houdini", "Wukong_gsplat.hip")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")
BRUSH_EXE = os.path.join(ROOT, "tools", "brush", "brush_app.exe")

# Character network inside the source hip
CREATE = "/stage/imp_Chr_Jiao_model_master_v0001/sopnet/create"

# Each part is trained separately so surfaces hidden by the other part
# (body under cloth) still get learned and stay correct when things move.
#   body  : bone-driven (Body + BackHalo), trained in the bind pose
#   cloth : velum-simulated (Cloth + Accessory), trained at REST_FRAME
PARTS = ("body", "cloth")
REST_FRAME = 1

# Dataset render settings
RESOLUTION = 1600
FOCAL_MM = 35.0
APERTURE_MM = 36.0
ORBIT_VIEWS = 180      # full-sphere views framing the whole part
CLOSE_VIEWS = 90       # closer views for fine detail (face, ornaments)
CLOSE_DISTANCE = 0.5   # fraction of orbit distance
SAMPLES = 64
ENGINE = "xpu"          # "cpu" if the GPU is busy / XPU misbehaves
CAM_START_FRAME = 1001

"""Step 3 of GSplat Studio: COLMAP on the orbit, optionally checked by Claude.

  tools/venv/Scripts/python scripts/gs_colmap_ai.py <name> [--circle] [--ai] [--rounds 3]

1. free COLMAP solve on the masked frames (orbit_solve angles). By default that
   solve is the result, as long as it placed at least 80% of the frames; it beat
   the circle clearly in a side-by-side training test (see orbit_solve.use_free).
Fallback, or always with --circle / --ai / --keep-angles:
2. put the cameras on a perfect circle at the measured angles, triangulate (solve)
3. review: reprojection per frame + labelled frame and overlay sheets
4. only with --ai (experimental, off by default):
   Claude (Claude Code CLI, headless, read-only) looks at the sheets and the
   numbers and answers with a corrected angle table if a frame sits at the wrong
   angle; this script checks that table, writes it, and the solve is redone -
   up to --rounds times, stopping as soon as Claude is satisfied.

Output: colmap/<name>/images + sparse/0; scratch and review in colmap/_work/<name>.

Why the Claude check is off: in a blind test (2026-10-22) it confirmed a correct
set but missed a deliberate 12-degree error, as did the reprojection numbers -
the points and the splat bend to fit whatever cameras they are given. The
accuracy comes from step 1, the measuring solve.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbit_solve as orbit  # noqa: E402
from video_to_dataset import progress  # noqa: E402

# Claude answers partly in Vietnamese; a piped stdout on Windows is cp1252 otherwise.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLAUDE = os.path.expanduser(r"~\.local\bin\claude.exe")
CLAUDE_MODEL = "opus"
CLAUDE_MAX_TURNS = 30
ANGLES_SPAN = (0, 45)   # progress: free solve; the rounds share 45..100

PROMPT = """You are checking the camera angles of a 360-degree orbit video (AI generated, \
character "{name}", {count} frames) before Gaussian-splat training. The camera moves on one \
level circle around the character; each frame was placed at the angle listed in orbit_keys.json \
(frame -> degrees, linear between keys; "sweep_deg" closes the loop). The angles came from a \
COLMAP solve, which can be off where the AI video turns unevenly.

Files in this folder:
- review/frames_sheet.jpg: 24 frames round the turn, each labelled "frame angle".
- review/overlay_sheet.jpg: 3D points projected back onto 8 frames (green = within 1.5 px, \
yellow = within 4 px, red = worse). Points sliding off the body mean the angle is wrong there.
- review/review.json: per frame: angle, points, error_px, wants_move_deg (how far that frame's \
own features pull it along the circle; > 1 degree is suspicious).
- orbit_keys.json: the current angle table.

Round {round} of {rounds}. Current numbers:
{table}

Check the angles visually: frame 1 is 0 degrees; at 90 and 270 you see a clean side profile; at \
180 the back; a T-pose arm span shrinks with |cos(angle)|. Mirror views (a and 360-a) should look \
like mirror images of each other. Compare the labels on frames_sheet.jpg with what you see.

You can only read files. If every frame looks right within about 2 degrees, keep the table. \
Otherwise give a corrected table: start from the current "keys" and change or add only the keys \
near the problem (frames as integers 1..{count}, angles in degrees).

End with exactly these lines:
SUMMARY: <one short sentence in Vietnamese for a non-technical artist: what you checked or changed>
KEYS: <only if you changed something: the full new keys list as one-line JSON, e.g. [[1, 0.0], [11, 14.3]]>
VERDICT: OK   (or VERDICT: CHANGED when you gave KEYS)"""


def table(review_json: str, every: int = 10) -> str:
    rows = json.load(open(review_json))["rows"]
    lines = ["frame  angle   points  error_px  wants_move"]
    for r in rows[::every]:
        lines.append(f"{r['frame']:5d}  {r['angle']:6.1f}  {r['points']:6d}  {r['error_px']:7.2f}  "
                     f"{r['wants_move_deg']:+6.2f}")
    return "\n".join(lines)


def ask_claude(p: dict, name: str, round_no: int, rounds: int) -> bool:
    """Run Claude on the review; True if it changed the angle table."""
    count = len(os.listdir(p["rgb"]))
    prompt = PROMPT.format(name=name, count=count, round=round_no, rounds=rounds,
                           table=table(os.path.join(p["review"], "review.json")))
    cmd = [CLAUDE, "-p", prompt, "--model", CLAUDE_MODEL, "--max-turns", str(CLAUDE_MAX_TURNS),
           "--allowedTools", "Read", "--output-format", "text"]
    proc = subprocess.Popen(cmd, cwd=p["work"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    new_keys = None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("KEYS:"):
            new_keys = line[5:].strip()
        elif line:
            print(f"[Claude] {line}", flush=True)
    if proc.wait() != 0:
        print("Claude did not finish - keeping the current angles")
        return False
    if not new_keys:
        return False
    keys = orbit.load_json(p["keys"])
    try:
        table_new = sorted([int(f), float(a)] for f, a in json.loads(new_keys))
        if not table_new or any(not 1 <= f <= count for f, _ in table_new):
            raise ValueError("frame out of range")
        if not all(math.isfinite(a) for _, a in table_new):
            raise ValueError("angle is not a number")
    except (ValueError, TypeError) as err:
        print(f"Claude's angle table is unusable ({err}) - keeping the current angles")
        return False
    if table_new == keys["keys"]:
        return False
    shutil.copy2(p["keys"], p["keys"] + f".round{round_no}.bak")
    keys["keys"] = table_new
    orbit.save_json(p["keys"], keys)
    print(f"[Claude] angle table updated: {len(table_new)} keys")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ai", action="store_true", help="let Claude review the angles (experimental)")
    ap.add_argument("--keep-angles", action="store_true", help="reuse orbit_keys.json, skip the free solve")
    # first guess only (35 mm equivalent, 36 mm sensor width); the solve refines it.
    # 35 = the lens of our playblast references, which the AI videos follow
    ap.add_argument("--focal-mm", type=float, default=35.0)
    ap.add_argument("--height", type=float, default=1.65)
    ap.add_argument("--keep-bg", action="store_true",
                    help="match features on the background too (textured set such as a checker floor)")
    ap.add_argument("--circle", action="store_true", help="always put the cameras on a perfect circle")
    args = ap.parse_args()
    p = orbit.paths(args.name)
    if not os.path.isdir(p["images"]):
        sys.exit(f"run step 2 first: {p['images']} is missing")

    if not (args.keep_angles and os.path.exists(p["keys"])):
        orbit.measure_angles(p, args.focal_mm, 360.0, use_masks=not args.keep_bg)
        # angle edits and the Claude review only mean something on the circle
        if not (args.circle or args.ai) and orbit.use_free(p, args.height):
            progress(100, "COLMAP xong")
            print("colmap ready:", p["out"])
            return
    rounds = max(1, args.rounds) if args.ai else 1
    span = (100 - ANGLES_SPAN[1]) / rounds
    for r in range(1, rounds + 1):
        base = ANGLES_SPAN[1] + span * (r - 1)
        progress(base, f"Dat camera len vong tron (vong {r})")
        orbit.solve(p, args.focal_mm, args.height)
        progress(base + span * 0.3, "Kiem tra tung anh")
        orbit.review(p)
        print("GSS_PREVIEW", os.path.join(p["review"], "overlay_sheet.jpg"), flush=True)
        if not args.ai:
            break
        progress(base + span * 0.4, f"Claude dang kiem tra (vong {r}/{rounds})")
        if not ask_claude(p, args.name, r, rounds):
            break
        if r == rounds:  # Claude changed something in the last round: apply it once more
            progress(97, "Dat lai camera theo goc Claude sua")
            orbit.solve(p, args.focal_mm, args.height)
            orbit.review(p)
    progress(100, "COLMAP xong")
    print("colmap ready:", p["out"])


if __name__ == "__main__":
    main()

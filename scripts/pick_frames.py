"""venv: pick a small set of sharp, well-spread frames from an orbit video for COLMAP + training.

  tools/venv/Scripts/python scripts/pick_frames.py <video.mp4> <id> [--min 0.3] [--max 0.5]

CEO (2026-10-29): more frames did not fix the splats - good frames from every angle do. Keep only frames with no
motion blur, each sharing about 30-50 % of its content with the previous pick (30 % is plenty for COLMAP), and no
matte. Overlap = area (convex hull) of the SIFT matches that survive a RANSAC fundamental-matrix fit, over the area
of all the frame's keypoints;
blur = Laplacian variance (gs_frames.sharpness) against the frames around it (a camera whip shows as a dip).
Walk: from the sharpest of the first frames, look ahead until the overlap drops under --max; every frame from there
until it drops under --min is a candidate, and the sharpest candidate is the next pick.
Out: chars/<id>/frames/0001.png ... (full size, renumbered), chars/<id>/picked.json, chars/<id>/picked_sheet.jpg.
"""
import argparse
import json
import os

import cv2
import numpy as np

ROOT = os.environ.get("GSPLAT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_W = 540          # analysis width (features and blur); full-size frames are written
SHARP_WIN = 15        # frames each side for the local sharpness reference
SHARP_MIN = 0.8       # a candidate under this share of its neighbourhood's median sharpness is blurred


def overlap(a: tuple, b: tuple, matcher: cv2.BFMatcher) -> float:
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0.0
    good = [m for m, n in (p for p in matcher.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0.0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.999)
    if inl is None or inl.sum() < 8:
        return 0.0
    # share of the frame's content area (hull of all its keypoints) that the other frame also shows (hull of the
    # matches): a pixel overlap. The match count alone read low - the flat blue screen matches poorly frame to frame
    shared = cv2.contourArea(cv2.convexHull(pa[inl.ravel() == 1]))
    content = cv2.contourArea(cv2.convexHull(np.float32([k.pt for k in ka])))
    return shared / max(content, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("id")
    ap.add_argument("--min", type=float, default=0.3)
    ap.add_argument("--max", type=float, default=0.5)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    full, gray = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        full.append(f)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray.append(cv2.resize(g, (WORK_W, round(g.shape[0] * WORK_W / g.shape[1])), interpolation=cv2.INTER_AREA))
    n = len(gray)
    sharp = np.array([cv2.Laplacian(g, cv2.CV_32F).var() for g in gray])
    ref = np.array([np.median(sharp[max(0, i - SHARP_WIN):i + SHARP_WIN + 1]) for i in range(n)])
    rel = sharp / ref
    sift = cv2.SIFT_create(3000)
    feats = [sift.detectAndCompute(g, None) for g in gray]
    matcher = cv2.BFMatcher()

    picks = [int(np.argmax(rel[:12]))]
    while True:
        cur, cands, j = picks[-1], [], picks[-1] + 1
        while j < n:
            o = overlap(feats[cur], feats[j], matcher)
            if o < args.min:
                break
            if o <= args.max:
                cands.append((j, o))
            j += 1
        if j >= n and not cands:
            break
        if not cands:   # the view changed faster than one frame can bridge: take the next frame anyway
            cands = [(cur + 1, 0.0)]
        sharp_ok = [c for c in cands if rel[c[0]] >= SHARP_MIN] or cands
        nxt = max(sharp_ok, key=lambda c: rel[c[0]])
        picks.append(nxt[0])
        print(f"pick {len(picks):3d}: frame {nxt[0] + 1:4d}  overlap {nxt[1]:.2f}  sharpness x{rel[nxt[0]]:.2f}")
        if j >= n:
            break

    char = os.path.join(ROOT, "chars", args.id)
    out = os.path.join(char, "frames")
    os.makedirs(out, exist_ok=True)
    for k, p in enumerate(picks, 1):
        cv2.imwrite(os.path.join(out, f"{k:04d}.png"), full[p])
    json.dump({"video": args.video, "frames_in_video": n, "picked": [p + 1 for p in picks],
               "overlap": [args.min, args.max]}, open(os.path.join(char, "picked.json"), "w"), indent=1)
    cols = 10
    tiles = [cv2.resize(full[p], (108, 192)) for p in picks]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(os.path.join(char, "picked_sheet.jpg"),
                np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]))
    print(f"{len(picks)} of {n} frames -> {out}")


if __name__ == "__main__":
    main()

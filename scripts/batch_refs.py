"""venv: reference photos of one chiLinh54 character, ready for Seedream / Seedance.

  tools/venv/Scripts/python scripts/batch_refs.py <id>          e.g. 001

Reads  <CHARS_SRC>/<id>_jpg/1..4.jpg (front, side, back, side; 3000x4000 on a blue screen).
Writes chars/<id>/refs/ref_1..4.jpg (2048 px tall, the face blurred so the generated portrait gives the face,
not the costume model) and chars/<id>/refs/sheet.jpg (all four side by side, for review).
The front crop chars/<id>/refs/front_upper_blur.jpg (head to waist, face blurred) is the Seedream reference.
Faces are found with OpenCV YuNet (tools/models/face_detection_yunet_2023mar.onnx); a side or back view
without a detected face is left as it is.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.environ.get("GSPLAT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARS_SRC = os.environ.get("CHARS_SRC", "")
YUNET = os.path.join(ROOT, "tools", "models", "face_detection_yunet_2023mar.onnx")
HEIGHT = 2048
FACE_PAD = 0.35          # blur box grows by this much on each side of the detected face


def faces(img: np.ndarray) -> list:
    det = cv2.FaceDetectorYN.create(YUNET, "", (img.shape[1], img.shape[0]), 0.6)
    _, found = det.detect(img)
    return [] if found is None else [tuple(int(v) for v in f[:4]) for f in found]


def blur_faces(img: np.ndarray) -> tuple[np.ndarray, int]:
    out = img.copy()
    boxes = faces(img)
    for x, y, w, h in boxes:
        px, py = int(w * FACE_PAD), int(h * FACE_PAD)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(img.shape[1], x + w + px), min(img.shape[0], y + h + py)
        k = max(31, (x1 - x0) // 3 | 1)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (k, k), 0)
    return out, len(boxes)


def main(cid: str) -> None:
    src = os.path.join(CHARS_SRC, f"{cid}_jpg")
    out = os.path.join(ROOT, "chars", cid, "refs")
    os.makedirs(out, exist_ok=True)
    small = []
    for i in (1, 2, 3, 4):
        img = cv2.imread(os.path.join(src, f"{i}.jpg"))
        img = cv2.resize(img, (round(img.shape[1] * HEIGHT / img.shape[0]), HEIGHT), interpolation=cv2.INTER_AREA)
        img, n = blur_faces(img)
        if i == 1:   # head to waist of the front view: Seedream copies the costume, the prompt asks for a new face
            cv2.imwrite(os.path.join(out, "front_upper_blur.jpg"), img[: HEIGHT * 55 // 100], [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"ref_{i}: {n} face(s) blurred")
        cv2.imwrite(os.path.join(out, f"ref_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        small.append(cv2.resize(img, None, fx=0.4, fy=0.4))
    cv2.imwrite(os.path.join(out, "sheet.jpg"), np.hstack(small), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("refs ->", out)


if __name__ == "__main__":
    main(sys.argv[1])

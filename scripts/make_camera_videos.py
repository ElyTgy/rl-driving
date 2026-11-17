"""Usage: python3 scripts/make_camera_videos.py raw-data

Build color.mp4 and depth.mp4 inside each run's camera/ folder.

Frames are placed by their filename timestamp on a fixed 15 fps timeline, so
dropped frames are held (video plays in real time).
Depth: raw uint16 (likely mm), clipped to [0, DEPTH_MAX], TURBO colormap,
invalid pixels (0 or 65535) drawn black.
"""
import glob, os, re, sys
import cv2
import numpy as np

FPS = 15.0
DEPTH_MAX = 10000.0
ROOT = sys.argv[1]


def ts(path):
    return float(re.search(r"_([\d.]+)s\.", os.path.basename(path)).group(1))


def open_writer(path, size):
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), FPS, size)
        if w.isOpened():
            return w, codec
    raise RuntimeError("no working mp4 codec")


def load_depth(path):
    with open(path) as f:
        a = np.array(f.read().split(), dtype=np.float32).reshape(480, 640)
    invalid = (a <= 0) | (a >= 65535)
    img = (np.clip(a / DEPTH_MAX, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    img[invalid] = 0
    return img


def write_stream(files, loader, out):
    files = sorted(files, key=ts)
    times = [ts(f) for f in files]
    first = loader(files[0])
    h, w = first.shape[:2]
    writer, codec = open_writer(out, (w, h))
    n_out = int(times[-1] * FPS) + 1
    idx, cur, cur_idx = 0, first, 0
    for k in range(n_out):
        t = k / FPS
        while idx + 1 < len(files) and times[idx + 1] <= t + 1e-6:
            idx += 1
        if idx != cur_idx:
            cur, cur_idx = loader(files[idx]), idx
        writer.write(cur)
    writer.release()
    return len(files), n_out, codec


for run in sorted(glob.glob(os.path.join(ROOT, "*run_*"))):
    cam = os.path.join(run, "camera")
    name = os.path.basename(run)
    if not os.path.isdir(cam):
        print(f"{name}: no camera folder, skipped")
        continue
    for stream, pattern, loader in (
        ("color", "*.png", cv2.imread),
        ("depth", "*.txt", load_depth),
    ):
        files = glob.glob(os.path.join(cam, stream, pattern))
        if len(files) < 2:
            print(f"{name}/{stream}: only {len(files)} frame(s), skipped")
            continue
        n_in, n_out, codec = write_stream(files, loader, os.path.join(cam, f"{stream}.mp4"))
        print(f"{name}/{stream}.mp4: {n_in} frames -> {n_out} @ {FPS:g} fps ({n_out/FPS:.1f}s, {codec})")

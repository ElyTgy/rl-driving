"""Generate a simulator map of a closed track that has only left turns (when driven counterclockwise).

Why: bc_v1 was trained on a loop of left turns, so it crashes at sharp right turns in the sim.
To test whether it generalises to a *different track* with the kind of turns it has seen, we need
a track that is not the ims_small oval but still has only left turns.

How the track is built:
    1. Take a convex polygon (CORE_POLYGON). Convex means every corner turns the same way, so
       driving around it counterclockwise gives only left turns.
    2. The centerline is that polygon pushed outward by CORNER_RADIUS: straight edges stay straight
       and every corner becomes a circular arc, so the corners have different angles but the
       same radius.
    3. Every pixel within TRACK_WIDTH / 2 of the centerline is free (white); everything else is
       wall (black). The edges of the free band are the inner and outer walls.

Output (same format as the maps in f1tenth_gym_ros/maps): ros/maps/left_loop.png + left_loop.yaml.
The script also prints a start pose for the sim config.

Usage (from the repo root): .venv/bin/python scripts/make_left_loop_map.py
"""
import os

import numpy as np
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_DIR, 'ros', 'maps')
NAME = 'left_loop'

# Track shape. Vertices in metres, counterclockwise order, must form a convex polygon.
CORE_POLYGON = np.array([(0.0, 0.0), (14.0, 0.0), (17.0, 6.0), (9.0, 11.0), (0.0, 8.0)])
CORNER_RADIUS = 1.7     # m, radius of the centerline in the corners (the outer wall is 0.9 m wider)
TRACK_WIDTH = 1.8       # m, wall to wall (ims_small is 2.0 m)

RESOLUTION = 0.05       # m per pixel
MARGIN = 1.0            # m of wall around the track in the image


def rounded_polygon(vertices, radius, points_per_corner=40):
    """Points along the outline of `vertices` pushed outward by `radius` (corners become arcs)."""
    points = []
    n = len(vertices)
    for i in range(n):
        prev_v, v, next_v = vertices[i - 1], vertices[i], vertices[(i + 1) % n]
        # Outward normals of the edge arriving at v and the edge leaving v (CCW polygon: right-hand side).
        angle_in = np.arctan2(*(v - prev_v)[::-1]) - np.pi / 2
        angle_out = np.arctan2(*(next_v - v)[::-1]) - np.pi / 2
        if angle_out < angle_in:
            angle_out += 2 * np.pi
        for a in np.linspace(angle_in, angle_out, points_per_corner):
            points.append(v + radius * np.array([np.cos(a), np.sin(a)]))
    return np.array(points)


def distance_to_closed_line(px, py, line):
    """Distance from each point (px, py) to the closed polyline `line` (N, 2)."""
    a, b = line, np.roll(line, -1, axis=0)                      # segments a[i] -> b[i]
    best = np.full(px.shape, np.inf)
    for (ax, ay), (bx, by) in zip(a, b):
        dx, dy = bx - ax, by - ay
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        best = np.minimum(best, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def is_convex_ccw(vertices):
    edges = np.roll(vertices, -1, axis=0) - vertices
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    return bool(np.all(cross > 0))


def main():
    if not is_convex_ccw(CORE_POLYGON):
        raise SystemExit('CORE_POLYGON must be convex and counterclockwise, otherwise the track has right turns')
    centerline = rounded_polygon(CORE_POLYGON, CORNER_RADIUS)

    # Image covers the track plus a margin. Origin = world position of the bottom-left corner.
    half = TRACK_WIDTH / 2 + MARGIN
    x_min, y_min = centerline.min(axis=0) - half
    x_max, y_max = centerline.max(axis=0) + half
    width, height = int(np.ceil((x_max - x_min) / RESOLUTION)), int(np.ceil((y_max - y_min) / RESOLUTION))

    # World coordinates of every pixel centre. Image row 0 is the top (largest y).
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    px = x_min + (cols + 0.5) * RESOLUTION
    py = y_min + (height - rows - 0.5) * RESOLUTION
    free = distance_to_closed_line(px, py, centerline) <= TRACK_WIDTH / 2

    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).save(os.path.join(OUT_DIR, f'{NAME}.png'))
    with open(os.path.join(OUT_DIR, f'{NAME}.yaml'), 'w') as f:
        f.write(f'image: {NAME}.png\n'
                f'resolution: {RESOLUTION}\n'
                f'origin: [{x_min:.3f}, {y_min:.3f}, 0.000000]\n'
                'negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196\n')

    # Start pose: middle of the first straight, on the centerline, heading along it (counterclockwise).
    v0, v1 = CORE_POLYGON[0], CORE_POLYGON[1]
    heading = np.arctan2(v1[1] - v0[1], v1[0] - v0[0])
    outward = np.array([np.sin(heading), -np.cos(heading)])
    start = (v0 + v1) / 2 + CORNER_RADIUS * outward
    lap = np.sum(np.linalg.norm(np.diff(centerline, axis=0, append=centerline[:1]), axis=1))
    print(f'saved {OUT_DIR}/{NAME}.png ({width}x{height} px) and {NAME}.yaml')
    print(f'centerline lap length {lap:.1f} m; start pose sx: {start[0]:.2f}, sy: {start[1]:.2f}, stheta: {heading:.4f}')


if __name__ == '__main__':
    main()

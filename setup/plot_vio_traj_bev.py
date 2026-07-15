#!/usr/bin/env python
"""Plot a pure-VIO bird's-eye scatter for quick trajectory diagnosis.

This script does not touch MASt3R outputs. It only reads:
  - datasets/<name>/vio.txt
  - datasets/<name>/timestamps.txt (optional, overlay front-camera sample times)
  - datasets/<name>/imu.txt (optional, rotate VIO world to Z-up for a cleaner BEV)

Typical usage:
  python setup/plot_vio_traj_bev.py
  python setup/plot_vio_traj_bev.py --dataset datasets/cfds_floor1
  python setup/plot_vio_traj_bev.py --dataset datasets/cfds_floor1 --no-gravity
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mast3r_slam.run_config import load_run_config, run_dir  # noqa: E402


def load_vio(path):
    vio = np.loadtxt(path, dtype=np.float64)
    if vio.ndim == 1:
        vio = vio[None]
    order = np.argsort(vio[:, 0], kind="stable")
    vio = vio[order]
    _, uniq = np.unique(vio[:, 0], return_index=True)
    return vio[np.sort(uniq)]


def load_rgb_times(path):
    ts = np.loadtxt(path, dtype=np.float64)
    if ts.ndim == 1:
        ts = ts[None]
    return ts[:, 1]


def gravity_to_zup(g_world):
    """Return a rotation that maps gravity to -Z, making the output Z-up."""
    z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    v = np.cross(g_world, z)
    s = np.linalg.norm(v)
    c = float(g_world @ z)
    if s < 1e-8:
        return np.eye(3, dtype=np.float64) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))


def estimate_gravity_rotation(vio, imu_path):
    imu = np.loadtxt(imu_path, dtype=np.float64)
    if imu.ndim == 1:
        imu = imu[None]
    gyr = np.linalg.norm(imu[:, 4:7], axis=1)
    acc = np.linalg.norm(imu[:, 1:4], axis=1)
    stat = (gyr < 0.03) & (acc > 9.4) & (acc < 10.2)
    if not np.any(stat):
        raise RuntimeError("no static IMU segment found for gravity estimation")

    vt = vio[:, 0]
    slerp = Slerp(vt, Rotation.from_quat(vio[:, 4:8]))
    its = imu[stat, 0]
    inside = (its >= vt[0]) & (its <= vt[-1])
    its = its[inside]
    g_body = imu[stat][inside][:, 1:4]
    if len(its) == 0:
        raise RuntimeError("static IMU samples fall outside the VIO time range")

    g_world = slerp(its).apply(g_body).mean(0)
    g_world /= np.linalg.norm(g_world)
    return gravity_to_zup(g_world), g_world


def interpolate_positions(times, vio):
    vt = vio[:, 0]
    vp = vio[:, 1:4]
    times = np.asarray(times, dtype=np.float64)
    keep = (times >= vt[0]) & (times <= vt[-1])
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64), keep
    out = np.stack([np.interp(times[keep], vt, vp[:, k]) for k in range(3)], axis=1)
    return out, keep


def fit_bev(points_xy, width, height, margin):
    mn = points_xy.min(0)
    mx = points_xy.max(0)
    ctr = (mn + mx) / 2.0
    half = max(float(mx[0] - mn[0]), float(mx[1] - mn[1])) * 0.55 + 1e-6

    def project(xy):
        px = (xy[:, 0] - (ctr[0] - half)) / (2.0 * half)
        py = (xy[:, 1] - (ctr[1] - half)) / (2.0 * half)
        px = margin + px * (width - 2 * margin)
        py = height - (margin + py * (height - 2 * margin))
        return np.stack([px, py], axis=1)

    return project, ctr, half


def draw_grid(img, ctr, half, project, step_m=2.0):
    if step_m <= 0:
        return
    h, w = img.shape[:2]
    xmin, xmax = ctr[0] - half, ctr[0] + half
    ymin, ymax = ctr[1] - half, ctr[1] + half
    xs = np.arange(np.floor(xmin / step_m) * step_m, xmax + step_m, step_m)
    ys = np.arange(np.floor(ymin / step_m) * step_m, ymax + step_m, step_m)

    for x in xs:
        p = project(np.array([[x, ymin], [x, ymax]], dtype=np.float64)).astype(np.int32)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (235, 235, 235), 1, cv2.LINE_AA)
    for y in ys:
        p = project(np.array([[xmin, y], [xmax, y]], dtype=np.float64)).astype(np.int32)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (235, 235, 235), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (220, 220, 220), 1, cv2.LINE_AA)


def draw_points(img, pts_px, color, radius):
    for x, y in np.round(pts_px).astype(np.int32):
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def draw_marker(img, pt, color, text):
    x, y = np.round(pt).astype(np.int32)
    cv2.circle(img, (int(x), int(y)), 8, color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), 10, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        text,
        (int(x) + 10, int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )


def main():
    rc = load_run_config()
    default_dataset = rc.get("dataset", "datasets/cfds_floor28")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=default_dataset, help="dataset directory with vio.txt")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1400)
    ap.add_argument("--margin", type=int, default=70)
    ap.add_argument("--no-gravity", action="store_true", help="skip IMU gravity alignment")
    ap.add_argument("--rgb-step", type=int, default=1, help="plot every Nth front-camera sample")
    args = ap.parse_args()

    ds = pathlib.Path(args.dataset)
    out = (
        pathlib.Path(args.out)
        if args.out is not None
        else pathlib.Path(run_dir(rc)) / f"{ds.stem}_vio_traj_bev.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    vio = load_vio(ds / "vio.txt")
    pos = vio[:, 1:4].copy()
    coord_note = "raw_xy"
    g_world = None
    Rg = None
    if not args.no_gravity and (ds / "imu.txt").exists():
        try:
            Rg, g_world = estimate_gravity_rotation(vio, ds / "imu.txt")
            pos = (Rg @ pos.T).T
            coord_note = "gravity_zup_xy"
        except Exception as exc:
            print(f"[warn] gravity alignment skipped: {exc}")

    rgb_pos = np.zeros((0, 3), dtype=np.float64)
    n_rgb_total = 0
    n_rgb_plotted = 0
    if (ds / "timestamps.txt").exists():
        rgb_times = load_rgb_times(ds / "timestamps.txt")
        rgb_pos, keep = interpolate_positions(rgb_times, vio)
        n_rgb_total = int(np.count_nonzero(keep))
        step = max(int(args.rgb_step), 1)
        rgb_pos = rgb_pos[::step]
        if Rg is not None:
            rgb_pos = (Rg @ rgb_pos.T).T
        n_rgb_plotted = len(rgb_pos)

    all_xy = pos[:, :2]
    ref_xy = rgb_pos[:, :2] if len(rgb_pos) else all_xy
    project, ctr, half = fit_bev(ref_xy if len(ref_xy) else all_xy, args.width, args.height, args.margin)

    img = np.full((args.height, args.width, 3), 252, dtype=np.uint8)
    draw_grid(img, ctr, half, project, step_m=2.0)

    vio_px = project(all_xy)
    draw_points(img, vio_px, (214, 139, 58), 2)

    if len(rgb_pos):
        rgb_px = project(rgb_pos[:, :2])
        draw_points(img, rgb_px, (42, 118, 232), 3)
    else:
        rgb_px = np.zeros((0, 2), dtype=np.float64)

    draw_marker(img, vio_px[0], (46, 180, 79), "start")
    draw_marker(img, vio_px[-1], (231, 76, 60), "end")

    path_len = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
    span = pos.max(0) - pos.min(0)
    title = f"Pure VIO BEV Scatter  |  {ds.stem}"
    cv2.putText(img, title, (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    lines = [
        f"coord: {coord_note}",
        f"vio points: {len(pos)}   path: {path_len:.1f} m",
        f"span xyz: {span[0]:.2f}, {span[1]:.2f}, {span[2]:.2f} m",
        f"rgb samples used: {n_rgb_plotted}/{n_rgb_total}" if n_rgb_total else "rgb samples: none",
    ]
    if g_world is not None:
        lines.append(f"gravity(world): [{g_world[0]:.3f}, {g_world[1]:.3f}, {g_world[2]:.3f}]")

    y = 72
    for line in lines:
        cv2.putText(img, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 50, 50), 2, cv2.LINE_AA)
        y += 28

    legend = [
        ((214, 139, 58), "raw vio"),
        ((42, 118, 232), "front-camera sample times"),
        ((46, 180, 79), "start"),
        ((231, 76, 60), "end"),
    ]
    lx = 28
    ly = args.height - 34
    for color, label in legend:
        cv2.circle(img, (lx, ly - 6), 6, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (lx + 14, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (50, 50, 50), 2, cv2.LINE_AA)
        lx += 220

    cv2.imwrite(str(out), img)
    print(f"[vio_bev] saved {out}")


if __name__ == "__main__":
    main()

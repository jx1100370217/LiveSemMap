#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 LoTIS 分段供人工检查建图记忆序列。

每个段 -> 一个文件夹(名标 起点→终点), 内含该段全部关键帧图(按段内顺序命名、
图上标 kf 号/段内序/起终点, 首帧绿框 START、末帧红框 END), 及 _info.txt(元信息+等效Hz)。
顶层 _INDEX.txt 总览所有段。

用法: python nav_web/export_seg_check.py --run logs/cfds_floor28_run --seq cfds_floor28
"""
import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CJK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def load_font(size):
    for p in _CJK_FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size), True
            except Exception:
                pass
    return ImageFont.load_default(), False


def safe(name):
    return "".join(c if c not in '/\\:*?"<>|\n\t' else "_" for c in str(name)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/cfds_floor28_run")
    ap.add_argument("--seq", default="cfds_floor28")
    ap.add_argument("--dataset", default=None, help="原始帧目录(含 <id:06d>.png); 默认 datasets/<seq>")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seg = json.load(open(os.path.join(args.run, f"{args.seq}_lotis_seg.json")))
    raw = seg.get("frame_source", "keyframe") == "raw"   # M4c: 段帧=原始帧
    thumbs = os.path.join(args.run, "web", "thumbs")
    dataset = args.dataset or os.path.join("datasets", args.seq)
    out = args.out or os.path.join(args.run, "seg_check")
    os.makedirs(out, exist_ok=True)

    def load_frame(gid):     # 段帧图: raw=datasets/<id:06d>.png, 旧版=thumbs/kf{gid}.jpg
        p = os.path.join(dataset, f"{gid:06d}.png") if raw else os.path.join(thumbs, f"kf{gid}.jpg")
        return (Image.open(p).convert("RGB"), p) if os.path.exists(p) else (None, p)

    # 时间戳(原始帧成像时刻), 用于标每段等效 Hz。raw 段的帧序号=原始帧id, 直接查 ts;
    # 旧版关键帧段需经 fid=frame_ids 映射。
    ts = fid = None
    try:
        tsm = []
        for ln in open(os.path.join(dataset, "timestamps.txt")):
            p = ln.split()
            if p and not p[0].startswith("#"):
                tsm.append(float(p[1]))
        ts = np.array(tsm)
        if not raw:
            z = np.load(os.path.join(args.run, f"{args.seq}_occupancy.npz"))
            fid = np.asarray(z["frame_ids"]).ravel().astype(int)
    except Exception as e:
        print("[warn] 无时间戳, 跳过等效Hz:", e)

    font, cjk = load_font(13)
    if not cjk:
        print("[warn] 未找到中文字体, 图上起终点名用文件夹名替代")

    fs_tag = "原始帧" if raw else "关键帧"
    segs = seg["segments"]
    idx_lines = [f"# LoTIS 分段检查  seq={args.seq}  段数={len(segs)}  frame_source={seg.get('frame_source')}  "
                 f"frame_wh={seg['frame_wh']}  crop={seg['crop']}  max_len={seg.get('max_len')}",
                 f"# 段帧={fs_tag}; 图上顶栏=帧号 #段内序/末序; 首帧绿框 START, 末帧红框 END", ""]

    for si, s in enumerate(segs):
        fi = s.get("frame_indices") or s.get("kf_indices")
        n = len(fi)
        fname = s.get("from_name") or s.get("from_cat") or f"n{s.get('node_from')}"
        tname = s.get("to_name") or s.get("to_cat") or f"n{s.get('node_to')}"
        label = f"拐弯@{s.get('name','')}" if s.get("type") == "turn" else f"{fname}→{tname}"
        folder = os.path.join(out, f"{si:02d}__{safe(s['key'])}__{safe(label)}")
        os.makedirs(folder, exist_ok=True)

        hz_str = ""
        if ts is not None:
            try:
                idxs = np.asarray(fi) if raw else fid[np.asarray(fi)]   # -> 原始帧id
                tk = ts[idxs]
                span = float(tk[-1] - tk[0])
                hz = (n - 1) / span if span > 0 else 0.0
                hz_str = f"  跨时={span:.1f}s  等效={hz:.2f}Hz"
            except Exception:
                pass

        info = [f"段 {si}: {s['key']}   类型={s.get('type')}   段帧={fs_tag}",
                f"起点: {fname}   ({s.get('from_cat','')})   node={s.get('node_from')}",
                f"终点: {tname}   ({s.get('to_cat','')})   node={s.get('node_to')}",
                f"帧数={n}   帧范围={fi[0]}..{fi[-1]}   "
                f"编码后seq_len={s.get('seq_len')}   part={s.get('part')}/{s.get('n_parts')}{hz_str}",
                f"frame_indices={fi}", ""]
        open(os.path.join(folder, "_info.txt"), "w").write("\n".join(info))
        idx_lines.append(f"[{si:02d}] {s['key']:18s} {label}   帧={n}  f{fi[0]}-{fi[-1]}{hz_str}")

        for i, gid in enumerate(fi):
            im, path = load_frame(gid)
            if im is None:
                print(f"[warn] 缺帧图 {path}, 跳过")
                continue
            d = ImageDraw.Draw(im)
            first, last = i == 0, i == n - 1
            if first:
                d.rectangle([0, 0, im.width - 1, im.height - 1], outline=(60, 230, 120), width=4)
            elif last:
                d.rectangle([0, 0, im.width - 1, im.height - 1], outline=(255, 90, 90), width=4)
            head = f"f{gid}  #{i}/{n - 1}"
            if first:
                head += f"  START {fname if cjk else ''}"
            if last:
                head += f"  END {tname if cjk else ''}"
            d.rectangle([0, 0, im.width, 17], fill=(0, 0, 0))
            d.text((3, 2), head, fill=(255, 255, 255), font=font)
            suf = "_START" if first else ("_END" if last else "")
            im.save(os.path.join(folder, f"{i:03d}_f{gid:05d}{suf}.jpg"), quality=88)

    open(os.path.join(out, "_INDEX.txt"), "w").write("\n".join(idx_lines))
    print(f"[done] {len(segs)} 段 -> {out}")
    print("\n".join(idx_lines))


if __name__ == "__main__":
    main()

"""HMSG 四层节点数据结构与序列化 (照抄 fsr_vln/memory/hmsg/graph 的字段与
ply+json 文件夹协议, 加载端与其 load_hmsg_graph 兼容)。

id 命名约定 (父子关系靠 id 前缀恢复):
  floor_id="0" / room_id="0_3" / object_id="0_3_17" / view_id="0_3_42"
"""
import json
import pathlib

import numpy as np

# 房间配色 (区分度优先, 深底可读; 超出循环复用)
ROOM_PALETTE = ("#4fc3f7", "#ff8a65", "#aed581", "#ba68c8", "#ffd54f", "#4db6ac",
                "#f06292", "#9575cd", "#81c784", "#ffb74d", "#64b5f6", "#e57373",
                "#dce775", "#7986cb", "#4dd0e1", "#ff8fa3", "#a1887f", "#90a4ae")


def _pcd(points, colors=None):
    import open3d as o3d
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    if colors is not None:
        p.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(colors, np.float64), 0, 1))
    return p


class Floor:
    def __init__(self, floor_id, name=""):
        self.floor_id = str(floor_id)
        self.name = name or f"floor_{floor_id}"
        self.pcd = None                # o3d 点云
        self.vertices = None           # AABB 8 顶点 (8,3)
        self.floor_height = 0.0        # 层高
        self.floor_zero_level = 0.0    # 层地面高度
        self.rooms = []

    def to_json(self):
        return {"floor_id": self.floor_id, "name": self.name,
                "vertices": np.asarray(self.vertices, float).tolist()
                if self.vertices is not None else None,
                "floor_height": float(self.floor_height),
                "floor_zero_level": float(self.floor_zero_level)}


class Room:
    def __init__(self, room_id, floor_id):
        self.room_id = str(room_id)    # "{floor}_{idx}"
        self.floor_id = str(floor_id)
        self.name = ""                 # 房间类型 (generate_room_names 赋值)
        self.category = ""
        self.pcd = None
        self.vertices = None           # 2D 占据点集 (N,2) 地面平面 (米)
        self.room_height = 0.0
        self.room_zero_level = 0.0
        self.embeddings = []           # KMeans 代表视图特征 (<=K, D)
        self.represent_images = []     # 代表帧全局帧 id
        self.sample_images = []        # 房间内全部采样帧 id
        self.clip_embeddings = []      # 与 sample_images 对应的全图 CLIP (N, D)
        self.objects = []
        self.views = []
        self.name_zh = ""              # [扩展] Qwen 中文区域名 (优先门牌/公司名)
        self.type_zh = ""              # [扩展] 中文房型
        self.summary_zh = ""           # [扩展] 一句话中文摘要

    def to_json(self):
        return {"room_id": self.room_id, "floor_id": self.floor_id,
                "name": self.name, "category": self.category,
                "name_zh": self.name_zh, "type_zh": self.type_zh,
                "summary_zh": self.summary_zh,
                "vertices": np.asarray(self.vertices, float).tolist()
                if self.vertices is not None else None,
                "room_height": float(self.room_height),
                "room_zero_level": float(self.room_zero_level),
                "embeddings": [np.asarray(e, float).tolist()
                               for e in self.embeddings],
                "represent_images": [int(i) for i in self.represent_images],
                "sample_images": [int(i) for i in self.sample_images],
                "clip_embeddings": [np.asarray(e, float).tolist()
                                    for e in self.clip_embeddings]}


class Object:
    def __init__(self, object_id, room_id):
        self.object_id = str(object_id)   # "{floor}_{room}_{idx}"
        self.room_id = str(room_id)
        self.name = ""                    # 词表标签 (CLIP 最近邻)
        self.gt_name = ""
        self.pcd = None
        self.vertices = None              # 点云地面 2D 投影 (N,2) (照抄原版构建时语义)
        self.embedding = None             # 实例级 CLIP (D,)
        self.view_ids = []                # 可见该物体的 view
        self.best_view_id = None          # 平均深度最小的 view
        self.name_zh = ""                 # [扩展] 中文类别名

    def to_json(self):
        return {"object_id": self.object_id, "room_id": self.room_id,
                "name": self.name, "gt_name": self.gt_name,
                "name_zh": self.name_zh,
                "embedding": np.asarray(self.embedding, float).tolist()
                if self.embedding is not None else None,
                "view_ids": list(self.view_ids),
                "best_view_id": self.best_view_id}


class View:
    def __init__(self, view_id, room_id, img_id, img_path):
        self.view_id = str(view_id)       # "{floor}_{room}_{idx}"
        self.room_id = str(room_id)
        self.img_id = int(img_id)         # 数据集全局帧号
        self.img_path = str(img_path)
        self.pose = None                  # 6-DoF c2w (4,4) — 论文 view 层几何属性
        self.object_ids = []
        self.text_discription = []        # 视图内物体 name 列表 (拼写照抄原版)
        self.vlm_description = ""         # [扩展] LiveSemMap 在线 Qwen 逐帧描述

    def to_json(self):
        return {"view_id": self.view_id, "room_id": self.room_id,
                "img_id": self.img_id, "img_path": self.img_path,
                "pose": np.asarray(self.pose, float).tolist()
                if self.pose is not None else None,
                "object_ids": list(self.object_ids),
                "text_discription": list(self.text_discription),
                "vlm_description": self.vlm_description}


class HMSGGraph:
    """HMSG 容器: 节点列表 + networkx 层级图 + 序列化。"""

    def __init__(self):
        import networkx as nx
        self.floors, self.rooms, self.objects, self.views = [], [], [], []
        self.full_pcd = None              # 全局降采样点云 (o3d)
        self.full_feats = None            # (N_points, D) 逐点 CLIP (float16)
        self.mask_feats = []              # (N_inst, D) 实例 CLIP
        self.graph = nx.Graph()
        self.graph.add_node(0, name="building", type="building")

    # ---- 图拓扑组装 (照抄 create_graph_new) ----
    def assemble(self):
        g = self.graph
        for f in self.floors:
            g.add_node(f"F{f.floor_id}", type="floor")
            g.add_edge(0, f"F{f.floor_id}")
        for r in self.rooms:
            g.add_node(f"R{r.room_id}", type="room")
            g.add_edge(f"F{r.floor_id}", f"R{r.room_id}")
        for o in self.objects:
            g.add_node(f"O{o.object_id}", type="object")
            g.add_edge(f"R{o.room_id}", f"O{o.object_id}")
        for v in self.views:
            g.add_node(f"V{v.view_id}", type="view")
            g.add_edge(f"R{v.room_id}", f"V{v.view_id}")
            for oid in v.object_ids:
                g.add_edge(f"V{v.view_id}", f"O{oid}")

    # ---- 序列化 (照抄 save_hmsg_graph 的文件夹协议) ----
    def save(self, save_dir):
        import open3d as o3d
        import torch
        root = pathlib.Path(save_dir)
        gdir = root / "graph"
        for sub in ("floors", "rooms", "objects", "views"):
            d = gdir / sub
            if d.exists():           # 清残留 (重复构建时旧节点文件会混入 load)
                for f in d.iterdir():
                    f.unlink()
            d.mkdir(parents=True, exist_ok=True)
        if self.full_pcd is not None:
            o3d.io.write_point_cloud(str(root / "full_pcd.ply"), self.full_pcd)
        if self.full_feats is not None:
            torch.save(torch.from_numpy(np.asarray(self.full_feats)),
                       root / "full_feats.pt")
        if len(self.mask_feats):
            torch.save(torch.from_numpy(np.asarray(self.mask_feats, np.float32)),
                       root / "mask_feats.pt")
        for f in self.floors:
            if f.pcd is not None:
                o3d.io.write_point_cloud(str(gdir / "floors" / f"{f.floor_id}.ply"),
                                         f.pcd)
            (gdir / "floors" / f"{f.floor_id}.json").write_text(
                json.dumps(f.to_json()))
        for r in self.rooms:
            if r.pcd is not None:
                o3d.io.write_point_cloud(str(gdir / "rooms" / f"{r.room_id}.ply"),
                                         r.pcd)
            (gdir / "rooms" / f"{r.room_id}.json").write_text(
                json.dumps(r.to_json()))
        for o in self.objects:
            if o.pcd is not None:
                o3d.io.write_point_cloud(
                    str(gdir / "objects" / f"{o.object_id}.ply"), o.pcd)
            (gdir / "objects" / f"{o.object_id}.json").write_text(
                json.dumps(o.to_json()))
        for v in self.views:
            (gdir / "views" / f"{v.view_id}.json").write_text(
                json.dumps(v.to_json(), ensure_ascii=False))
        print(f"[hmsg] 已保存: {len(self.floors)}楼层/{len(self.rooms)}房间/"
              f"{len(self.objects)}物体/{len(self.views)}视图 -> {root}")

    @classmethod
    def load(cls, save_dir):
        import open3d as o3d
        root = pathlib.Path(save_dir)
        gdir = root / "graph"
        g = cls()
        if (root / "full_pcd.ply").exists():
            g.full_pcd = o3d.io.read_point_cloud(str(root / "full_pcd.ply"))
        for jf in sorted((gdir / "floors").glob("*.json")):
            d = json.loads(jf.read_text())
            f = Floor(d["floor_id"], d["name"])
            f.vertices = d["vertices"]
            f.floor_height, f.floor_zero_level = d["floor_height"], d["floor_zero_level"]
            ply = jf.with_suffix(".ply")
            if ply.exists():
                f.pcd = o3d.io.read_point_cloud(str(ply))
            g.floors.append(f)
        for jf in sorted((gdir / "rooms").glob("*.json")):
            d = json.loads(jf.read_text())
            r = Room(d["room_id"], d["floor_id"])
            r.name, r.category = d["name"], d.get("category", "")
            r.name_zh = d.get("name_zh", "")
            r.type_zh = d.get("type_zh", "")
            r.summary_zh = d.get("summary_zh", "")
            r.vertices = np.asarray(d["vertices"]) if d["vertices"] else None
            r.room_height, r.room_zero_level = d["room_height"], d["room_zero_level"]
            r.embeddings = [np.asarray(e, np.float32) for e in d["embeddings"]]
            r.represent_images = d["represent_images"]
            r.sample_images = d["sample_images"]
            r.clip_embeddings = [np.asarray(e, np.float32)
                                 for e in d["clip_embeddings"]]
            ply = jf.with_suffix(".ply")
            if ply.exists():
                r.pcd = o3d.io.read_point_cloud(str(ply))
            g.rooms.append(r)
        for jf in sorted((gdir / "objects").glob("*.json")):
            d = json.loads(jf.read_text())
            o = Object(d["object_id"], d["room_id"])
            o.name, o.gt_name = d["name"], d.get("gt_name", "")
            o.name_zh = d.get("name_zh", "")
            o.embedding = (np.asarray(d["embedding"], np.float32)
                           if d["embedding"] else None)
            o.view_ids, o.best_view_id = d["view_ids"], d["best_view_id"]
            ply = jf.with_suffix(".ply")
            if ply.exists():
                o.pcd = o3d.io.read_point_cloud(str(ply))
            g.objects.append(o)
        for jf in sorted((gdir / "views").glob("*.json")):
            d = json.loads(jf.read_text())
            v = View(d["view_id"], d["room_id"], d["img_id"], d["img_path"])
            v.pose = np.asarray(d["pose"]) if d.get("pose") else None
            v.object_ids = d["object_ids"]
            v.text_discription = d["text_discription"]
            v.vlm_description = d.get("vlm_description", "")
            g.views.append(v)
        g.assemble()
        return g

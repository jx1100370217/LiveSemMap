"""语义关键帧标注: 增量建图时对每个新关键帧异步调 VLM(L40 vLLM) 打语义标签,
并把相邻同类标注聚合成"语义节点"(门口/电梯间/茶水间等), 供 BEV 高亮与自然语言导航用。

- 标注在主进程内的线程池执行(HTTP IO 密集, 不占 GPU/不阻塞 SLAM 主循环);
- 结果写 mp.Manager().dict() {kf_idx: annotation}, viewer 进程可直接读;
- 节点位置不在此处存死: 聚合时由调用方传入各关键帧当前位置(VIO 或 SLAM 位姿),
  全局优化/尺度对齐后节点位置自动跟随。
"""
import base64
import json
import queue
import threading

import cv2
import numpy as np

# 类别表: category -> (中文名, BEV 颜色 RGB 0-1, 是否默认算地标)
# 参考 GIST/MapNav 等语义拓扑建图工作: 决策点(门口/路口) + 功能区 POI + 背景类
SEMANTIC_CATEGORIES = {
    "doorway":      ("门口",   (0.95, 0.55, 0.15), True),
    "junction":     ("路口",   (1.00, 0.85, 0.10), True),
    "elevator":     ("电梯间", (0.90, 0.15, 0.60), True),
    "stairs":       ("楼梯间", (0.60, 0.30, 0.90), True),
    "kitchen":      ("茶水间", (0.10, 0.85, 0.45), True),
    "restroom":     ("卫生间", (0.15, 0.60, 0.95), True),
    "meeting_room": ("会议室", (0.95, 0.30, 0.30), True),
    "office":       ("办公区", (0.55, 0.75, 0.95), False),
    "lobby":        ("大厅",   (0.95, 0.75, 0.55), True),
    "print_area":   ("打印区", (0.45, 0.85, 0.85), True),
    "lounge":       ("休息区", (0.75, 0.90, 0.35), True),
    "entrance":     ("出入口", (1.00, 0.40, 0.00), True),
    "corridor":     ("走廊",   (0.60, 0.60, 0.60), False),
    "other":        ("其他",   (0.50, 0.50, 0.50), False),
}

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(SEMANTIC_CATEGORIES.keys())},
        "name": {"type": "string", "maxLength": 24},
        "description": {"type": "string", "maxLength": 60},
        "confidence": {"type": "number"},
        "landmark": {"type": "boolean"},
    },
    "required": ["category", "name", "description", "confidence", "landmark"],
}

PROMPT = """你是室内导航语义地标标注员。观察这张第一人称视角图像，判断拍摄者当前所处的位置类型。

类别(category)必须从以下列表中选一个：
- doorway: 门口/门附近
- junction: 走廊路口/岔路口/转角
- elevator: 电梯间/电梯门
- stairs: 楼梯间/楼梯
- kitchen: 茶水间/水吧/餐区
- restroom: 卫生间(门口)
- meeting_room: 会议室(内部或门口可见)
- office: 办公区/工位区
- lobby: 大厅/前台/接待区
- print_area: 打印区/文印区
- lounge: 休息区/沙发区
- entrance: 建筑物或楼层出入口
- corridor: 普通走廊(无特殊地标)
- other: 其他/看不清

只输出一个 JSON 对象：
{"category": "类别", "name": "4-8字中文短名(如: 东侧茶水间)", "description": "一句话中文场景描述", "confidence": 0.0到1.0, "landmark": true或false}

规则：
- landmark=true 表示此处是值得作为导航地标的独特位置；corridor/office 中普通无特征的位置为 false。
- 图像朝向天花板/地面、严重模糊或看不清时: category=other, confidence<=0.3, landmark=false。
- name 要具体(如"玻璃门口""双开电梯间"), 不要重复类别泛称。"""


class SemanticAnnotator:
    """异步语义标注器: submit() 入队 -> 线程池调 vLLM -> 结果写共享 dict。"""

    def __init__(self, api_url, shared_ann, model="qwen3.5-9b", workers=3, timeout=30):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.ann = shared_ann          # mp.Manager().dict: kf_idx -> annotation dict
        self.timeout = timeout
        self.q = queue.Queue()
        self.submitted = set()         # 已提交过的 kf_idx (追赶式提交去重)
        self.fail_streak = 0
        self.disabled = False
        self.threads = [
            threading.Thread(target=self._worker, daemon=True) for _ in range(workers)
        ]
        for t in self.threads:
            t.start()

    def submit(self, kf_idx, frame_id, uimg):
        """主循环调用: uimg 为 CPU tensor/ndarray H×W×3 float RGB 0-1。仅做 uint8 拷贝+入队。"""
        if self.disabled or kf_idx in self.submitted:
            return
        self.submitted.add(kf_idx)
        img = np.asarray(uimg.cpu().numpy() if hasattr(uimg, "cpu") else uimg)
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        self.q.put((kf_idx, int(frame_id), img_u8))

    def catch_up(self, keyframes):
        """追赶式提交: 把 keyframes 容器里所有还没提交的关键帧补交
        (统一覆盖 INIT/TRACKING/backend重定位 三种 append 来源)。"""
        n = len(keyframes)
        for i in range(n):
            if i not in self.submitted:
                with keyframes.lock:
                    uimg = keyframes.uimg[i].clone()
                    fid = int(keyframes.dataset_idx[i])
                self.submit(i, fid, uimg)

    def pending(self):
        return self.q.qsize()

    def reset(self):
        """重新建图: 清空已提交集合/结果/未处理队列 (kf_idx 会从 0 复用)。"""
        try:
            while True:
                self.q.get_nowait()
                self.q.task_done()
        except queue.Empty:
            pass
        self.submitted.clear()
        self.ann.clear()

    def drain(self, print_progress=True):
        """退出前排空标注队列(保证已建图的关键帧都拿到语义标注)。"""
        import time
        while self.q.unfinished_tasks > 0 and not self.disabled:
            if print_progress:
                print(f"[semantic] 等待剩余 ~{self.q.qsize()} 个关键帧标注完成...")
            time.sleep(2.0)

    def _worker(self):
        while True:
            kf_idx, frame_id, img_u8 = self.q.get()
            if self.disabled:
                self.q.task_done()
                continue
            try:
                ann = self._annotate(img_u8)
                if ann is not None:
                    ann["kf_idx"] = kf_idx
                    ann["frame_id"] = frame_id
                    self.ann[kf_idx] = ann
                    self.fail_streak = 0
                    cat = ann.get("category", "?")
                    if SEMANTIC_CATEGORIES.get(cat, ("", 0, False))[2] and ann.get("landmark"):
                        print(f"[semantic] kf{kf_idx} (frame {frame_id}): "
                              f"{ann.get('name','')} [{cat}] conf={ann.get('confidence',0):.2f}")
                else:
                    self.fail_streak += 1
            except Exception as e:
                self.fail_streak += 1
                if self.fail_streak in (1, 10):
                    print(f"[semantic] 标注失败(kf{kf_idx}): {e}")
                if self.fail_streak >= 30:
                    if not self.disabled:
                        print("[semantic] 连续失败过多, 停用语义标注(建图不受影响)")
                    self.disabled = True
            finally:
                self.q.task_done()

    def _annotate(self, img_u8_rgb, retries=2):
        import requests

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_u8_rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "max_tokens": 300,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "annotation",
                                                "schema": ANNOTATION_SCHEMA}},
        }
        last_err = None
        for _ in range(retries + 1):
            try:
                r = requests.post(f"{self.api_url}/chat/completions",
                                  json=payload, timeout=self.timeout)
                r.raise_for_status()
                ann = json.loads(r.json()["choices"][0]["message"]["content"])
                if ann.get("category") not in SEMANTIC_CATEGORIES:
                    ann["category"] = "other"
                ann["confidence"] = float(np.clip(ann.get("confidence", 0.0), 0.0, 1.0))
                return ann
            except Exception as e:
                last_err = e
        raise last_err


def is_landmark(ann, min_conf=0.5):
    """一个标注是否算导航地标: VLM 判 landmark + 类别默认地标 + 置信度够。"""
    if ann is None:
        return False
    cat = ann.get("category", "other")
    default_lm = SEMANTIC_CATEGORIES.get(cat, ("", 0, False))[2]
    return bool(ann.get("landmark")) and default_lm and ann.get("confidence", 0) >= min_conf


def aggregate_nodes(ann_by_kf, pos_by_kf, merge_dist=2.5, min_conf=0.5):
    """把逐关键帧标注聚合成语义节点列表。

    ann_by_kf: {kf_idx: annotation dict}   (可直接传 Manager.dict 的快照)
    pos_by_kf: {kf_idx: (x, y, z) 世界系位置}  (调用方决定 VIO/SLAM 坐标)
    规则: 按 kf_idx 序把地标标注分段(同类别且相邻空间距离<merge_dist 连成段),
          段内取 confidence 最高者为代表; 再跨段做同类近距合并(走回头路去重)。
    返回: [{category, name, description, confidence, kf_indices, rep_kf, position}]
    """
    items = []
    for k in sorted(ann_by_kf.keys()):
        ann = ann_by_kf[k]
        if k in pos_by_kf and is_landmark(ann, min_conf):
            items.append((k, ann, np.asarray(pos_by_kf[k], np.float64)))
    if not items:
        return []

    # 1) 顺序分段: 同类且与段内最后一帧距离 < merge_dist
    segs = []
    for k, ann, p in items:
        cat = ann["category"]
        if segs and segs[-1]["category"] == cat and \
                np.linalg.norm(p - segs[-1]["positions"][-1]) < merge_dist:
            s = segs[-1]
            s["kf_indices"].append(k)
            s["positions"].append(p)
            s["anns"].append(ann)
        else:
            segs.append({"category": cat, "kf_indices": [k],
                         "positions": [p], "anns": [ann]})

    # 2) 跨段合并: 同类且中心距 < merge_dist (走回头路/多次经过同一地标)
    merged = []
    for s in segs:
        c = np.mean(s["positions"], axis=0)
        hit = None
        for m in merged:
            if m["category"] == s["category"] and \
                    np.linalg.norm(c - np.mean(m["positions"], axis=0)) < merge_dist:
                hit = m
                break
        if hit is not None:
            hit["kf_indices"] += s["kf_indices"]
            hit["positions"] += s["positions"]
            hit["anns"] += s["anns"]
        else:
            merged.append(s)

    # 3) 每组选置信度最高的标注为代表
    nodes = []
    for m in merged:
        best = int(np.argmax([a.get("confidence", 0) for a in m["anns"]]))
        rep = m["anns"][best]
        nodes.append({
            "category": m["category"],
            "name": rep.get("name", ""),
            "description": rep.get("description", ""),
            "confidence": float(rep.get("confidence", 0)),
            "kf_indices": [int(k) for k in m["kf_indices"]],
            "rep_kf": int(m["kf_indices"][best]),
            "position": [float(x) for x in np.mean(m["positions"], axis=0)],
        })
    return nodes

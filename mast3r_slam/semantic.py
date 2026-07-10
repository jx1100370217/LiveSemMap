"""语义关键帧标注: 增量建图时对每个新关键帧异步调 VLM(L40 vLLM) 打语义标签,
并把相邻同类标注聚合成"语义节点"(门口/电梯间/茶水间等), 供 BEV 高亮与自然语言导航用。

- 输入为该帧的 4 张环视鱼眼图 (insight9 camera_1..4 ≈ 前/右/后/左), 由前处理放在
  datasets/<name>/surround/{frame_id:06d}_{1..4}.jpg, 按 frame_id 从磁盘读取直接发 VLM;
- 标注在主进程内的线程池执行(HTTP IO 密集, 不占 GPU/不阻塞 SLAM 主循环);
- 结果写 mp.Manager().dict() {kf_idx: annotation}, viewer 进程可直接读;
- 节点位置不在此处存死: 聚合时由调用方传入各关键帧当前位置(VIO 或 SLAM 位姿),
  全局优化/尺度对齐后节点位置自动跟随。
"""
import base64
import json
import pathlib
import queue
import threading

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
        "signage": {"type": "array", "maxItems": 6,
                    "items": {"type": "string", "maxLength": 24}},
    },
    "required": ["category", "name", "description", "confidence", "landmark",
                 "signage"],
}

SURROUND_VIEWS = ("前", "右", "后", "左")  # camera_1..4 的大致朝向

PROMPT = """你是室内导航语义地标标注员。第 1 张图是巡检机器人的前视高清相机画面（分辨率最高，文字最清楚），后 4 张是同一时刻的环视鱼眼图，依次大致朝向机器人的前方、右侧、后方、左侧（图中可能拍到机器人自身部件和跟随的操作人员，忽略它们）。请综合所有画面，为机器人当前位置打一个对导航最有用的语义标签。

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

判定优先级（重要，从高到低）：
1. 画面中可见清晰可读的**专属名称**文字——公司名/门头/logo（如"翰德人力集团""晟和新能源"）、房间门牌号（如"2803""2807"）、电梯编号(A/B/C/D 或数字)、洗手间男女标识——**且该标识对应的门/设施就在机器人身旁**（判断依据：标识文字在画面中较大、位于正前方或侧旁很近处，机器人正处于其门口/跟前，通常 3 米内）：category 取对应类别（公司门头 -> lobby 或 doorway；电梯编号 -> elevator；洗手间 -> restroom 等），**name 直接用识别出的文字本身**，并把该处清楚读出的专属名称写进 signage。
   - 距离约束：沿走廊**远远看到**的门头/招牌（文字在画面中较小、位于通道远端，机器人还没走到跟前）**不算本帧地标**——此时按机器人实际所处环境标注（通常 corridor），signage 也不要写远处的标识。同一个门头只应在机器人真正经过它跟前的那几帧被标注。
   - 联合命名：同一处若同时可见房间编号和单位名称（如门牌"2807"与公司名"PHYMI"），name 用联合形式"2807-PHYMI"，signage 把两者分别写入。
   - "安全出口"、禁烟标志、方向箭头这类**通用指示牌不算**，不要写入 signage、更不要作为 name。
2. 无可读专属名称但可见明确的功能区地标——电梯门/呼梯面板、楼梯间、茶水间(水吧台/冰箱)、会议室(含透过玻璃可见的)、打印设备、前台/接待台、出入口、休息区沙发——取该类别，name 用简洁的设施名(如"玻璃会议室""茶水间""打印区")。
3. 身处两条以上走廊的交汇口/明显转角 -> junction。
4. 四周均为普通走廊墙面 -> corridor；均为普通办公工位 -> office。

只输出一个 JSON 对象：
{"category": "类别", "name": "节点名称(优先用画面专属文字, 2-12字)", "description": "一句话中文描述(地标在哪个方位、有什么特征)", "confidence": 0.0到1.0, "landmark": true或false, "signage": ["画面中清楚读出的专属名称文字"]}

规则：
- name 是干净的节点名称，**不要带"左侧/右侧/前方"等方位词**（方位写进 description）。
- signage 只写确实清晰可读的专属名称，读不清不要猜；没有就给空数组 []。
- 电梯若能看到编号(如 A/B/C/D)，name 必须带编号(如"A电梯")；看不到编号才用"电梯间"。
- landmark=true 对应优先级 1/2/3；优先级 4 的 corridor/office 为 false。
- 全部画面严重模糊或看不清时: category=other, confidence<=0.3, landmark=false。"""


class SemanticAnnotator:
    """异步语义标注器: submit() 入队 -> 线程池按 frame_id 读 4 环视图调 vLLM -> 结果写共享 dict。"""

    def __init__(self, api_url, shared_ann, surround_dir,
                 model="qwen3.5-35b-a3b", workers=8, timeout=60):
        # workers=8 配合 vLLM --max-num-seqs 8; 4 图/请求比单图慢, 但 insight9 新数据
        # 关键帧生成率低(帧间隔~2s), 标注可实时跟上建图。
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.ann = shared_ann          # mp.Manager().dict: kf_idx -> annotation dict
        self.surround_dir = pathlib.Path(surround_dir)  # {frame_id:06d}_{1..4}.jpg
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

    def surround_paths(self, frame_id):
        """该帧的语义图路径: _0=前视高清(读文字标识), _1.._4=环视鱼眼 (允许个别缺失)。"""
        paths = [self.surround_dir / f"{int(frame_id):06d}_{k}.jpg" for k in range(0, 5)]
        return [p for p in paths if p.exists()]

    def submit(self, kf_idx, frame_id):
        """主循环调用: 仅入队 (kf_idx, frame_id), 环视图由 worker 线程从磁盘读。"""
        if self.disabled or kf_idx in self.submitted:
            return
        self.submitted.add(kf_idx)
        self.q.put((kf_idx, int(frame_id)))

    def catch_up(self, keyframes):
        """追赶式提交: 把 keyframes 容器里所有还没提交的关键帧补交
        (统一覆盖 INIT/TRACKING/backend重定位 三种 append 来源)。"""
        n = len(keyframes)
        for i in range(n):
            if i not in self.submitted:
                with keyframes.lock:
                    fid = int(keyframes.dataset_idx[i])
                self.submit(i, fid)

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
            kf_idx, frame_id = self.q.get()
            if self.disabled:
                self.q.task_done()
                continue
            try:
                ann = self._annotate(self.surround_paths(frame_id))
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

    def _annotate(self, img_paths, retries=2):
        """img_paths: 该帧的环视 jpg 路径列表 (camera_1..4 顺序, 允许缺个别)。"""
        import requests

        if not img_paths:
            return None
        content = []
        for p in img_paths:
            b64 = base64.b64encode(p.read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": PROMPT})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
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
                # 内网 vLLM 直连, 显式绕过 shell 的 http(s)_proxy (走代理必失败)
                r = requests.post(f"{self.api_url}/chat/completions",
                                  json=payload, timeout=self.timeout,
                                  proxies={"http": None, "https": None})
                r.raise_for_status()
                ann = json.loads(r.json()["choices"][0]["message"]["content"])
                if ann.get("category") not in SEMANTIC_CATEGORIES:
                    ann["category"] = "other"
                ann["confidence"] = float(np.clip(ann.get("confidence", 0.0), 0.0, 1.0))
                ann["signage"] = [s.strip() for s in (ann.get("signage") or [])
                                  if isinstance(s, str) and s.strip()]
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


def _sigs(ann):
    """标注中的文字标识集合 (规范化)。"""
    return {s.strip() for s in ann.get("signage", []) if s and s.strip()}


def _sig_compatible(sigs_a, sigs_b):
    """文字标识兼容: 任一方没读到文字, 或读到的文字有交集。两方都有文字且完全
    不同(如 'A电梯' vs 'B电梯') -> 是不同地标, 不能合并。"""
    return not sigs_a or not sigs_b or bool(sigs_a & sigs_b)


def aggregate_nodes(ann_by_kf, pos_by_kf, merge_dist=2.5, min_conf=0.5):
    """把逐关键帧标注聚合成语义节点列表。

    ann_by_kf: {kf_idx: annotation dict}   (可直接传 Manager.dict 的快照)
    pos_by_kf: {kf_idx: (x, y, z) 世界系位置}  (调用方决定 VIO/SLAM 坐标)
    规则: 按 kf_idx 序把地标标注分段(同类别、相邻空间距离<merge_dist 且文字标识
          兼容才连成段 —— A电梯/B电梯这类编号不同的近邻地标不合并);
          段内代表优先取"读到文字标识"里置信度最高者(节点名尽量用真实门牌/公司名);
          再跨段做同类近距且标识兼容的合并(走回头路去重)。
    返回: [{category, name, description, confidence, kf_indices, rep_kf, position}]
    """
    items = []
    for k in sorted(ann_by_kf.keys()):
        ann = ann_by_kf[k]
        if k in pos_by_kf and is_landmark(ann, min_conf):
            items.append((k, ann, np.asarray(pos_by_kf[k], np.float64)))
    if not items:
        return []

    # 1) 顺序分段: 同类、与段内最后一帧距离 < merge_dist 且文字标识兼容
    segs = []
    for k, ann, p in items:
        cat = ann["category"]
        sig = _sigs(ann)
        s = segs[-1] if segs else None
        if s is not None and s["category"] == cat and \
                np.linalg.norm(p - s["positions"][-1]) < merge_dist and \
                _sig_compatible(s["sigs"], sig):
            s["kf_indices"].append(k)
            s["positions"].append(p)
            s["anns"].append(ann)
            s["sigs"] |= sig
        else:
            segs.append({"category": cat, "kf_indices": [k],
                         "positions": [p], "anns": [ann], "sigs": set(sig)})

    # 2) 跨段合并: 同类且 (a) 中心距 < merge_dist 且文字标识兼容 (走回头路/多次经过),
    #    或 (b) 专属文字标识有交集 (同名同类=同一地标, 不受距离限制 —— 电梯厅
    #    两侧/多次远近经过造成的同名重复段)
    merged = []
    for s in segs:
        c = np.mean(s["positions"], axis=0)
        hit = None
        for m in merged:
            if m["category"] != s["category"]:
                continue
            near = np.linalg.norm(c - np.mean(m["positions"], axis=0)) < merge_dist
            if (near and _sig_compatible(m["sigs"], s["sigs"])) or \
                    (m["sigs"] & s["sigs"]):
                hit = m
                break
        if hit is not None:
            hit["kf_indices"] += s["kf_indices"]
            hit["positions"] += s["positions"]
            hit["anns"] += s["anns"]
            hit["sigs"] |= s["sigs"]
        else:
            merged.append(s)

    # 3) 每组选代表: 读到文字标识的优先 (节点名用真实门牌/公司名), 再比置信度
    nodes = []
    for m in merged:
        scores = [(bool(_sigs(a)), a.get("confidence", 0)) for a in m["anns"]]
        best = int(max(range(len(scores)), key=lambda i: scores[i]))
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

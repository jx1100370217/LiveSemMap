"""语义关键帧标注: 增量建图时对每个新关键帧异步调 VLM(L40 vLLM) 打语义标签,
并把相邻同类标注聚合成"语义节点"(门口/电梯间/茶水间等), 供 BEV 高亮与自然语言导航用。

- 输入为该帧的前视高清图(_0) + 4 张环视鱼眼(_1.._4 ≈ 前/右/后/左), 由前处理放在
  datasets/<name>/surround/{frame_id:06d}_{0..4}.jpg, 按 frame_id 从磁盘读取直接发 VLM;
- 标注在主进程内的线程池执行(HTTP IO 密集, 不占 GPU/不阻塞 SLAM 主循环);
- 结果写 mp.Manager().dict() {kf_idx: annotation}, viewer 进程可直接读;
- 节点位置不在此处存死: 聚合时由调用方传入各关键帧当前位置(VIO 或 SLAM 位姿),
  全局优化/尺度对齐后节点位置自动跟随。
"""
import base64
import difflib
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
                 model="qwen3.5-35b-a3b", workers=8, timeout=60, min_dist=0.4):
        # workers=8 配合 vLLM --max-num-seqs 8; 多图/请求比单图慢, 但 insight9 新数据
        # 关键帧生成率低(帧间隔~2s), 标注可实时跟上建图。
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.ann = shared_ann          # mp.Manager().dict: kf_idx -> annotation dict
        self.surround_dir = pathlib.Path(surround_dir)  # {frame_id:06d}_{1..4}.jpg
        self.timeout = timeout
        # 空间抽稀: 距上一提交帧位移 < min_dist(米) 的关键帧不标注 (0=关闭)。
        # VIO 0.4m/kf 建帧下 D=0.4m 约筛掉 1/3 帧; 实测(cfds_floor28, 559kf)
        # 聚合节点除同门头类别变体外无损, D>=0.5 开始丢真节点 —— 勿轻易调大。
        # max_gap: 停留保护 —— 位移不足但已连续跳过 >=max_gap 帧时强制标 1 帧。
        # 驻足处常是地标(操作员停下看门头/等电梯), 纯距离抽稀会把停留段砍到
        # 只剩进入帧, 单帧 VLM 波动即丢节点(实测丢过 2803电梯)。
        self.min_dist = min_dist
        self.max_gap = 5
        self._last_pos = None          # 上一提交帧位置 (与 min_dist 同尺度)
        self._last_idx = None          # 上一提交帧 kf_idx (停留保护计数)
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
        """该帧的语义图路径: _0=前视高清(读文字标识), _1.._4=环视鱼眼 (允许个别缺失)。
        (实验过只发 _0/_2/_4 三图: 仅快 ~10%, 但帧级文字召回 93%->80%,
        且只在前/后鱼眼出现的门头会整个丢失, 故保留 5 图。)"""
        paths = [self.surround_dir / f"{int(frame_id):06d}_{k}.jpg" for k in range(0, 5)]
        return [p for p in paths if p.exists()]

    def submit(self, kf_idx, frame_id):
        """主循环调用: 仅入队 (kf_idx, frame_id), 环视图由 worker 线程从磁盘读。"""
        if self.disabled or kf_idx in self.submitted:
            return
        self.submitted.add(kf_idx)
        self.q.put((kf_idx, int(frame_id)))

    def submit_thinned(self, kf_idx, frame_id, pos):
        """带空间抽稀的提交 (在线/离线共用入口): pos 为该帧位置, None=不抽稀。
        跳过条件: 距上一提交帧位移 < min_dist 且连续跳过不足 max_gap 帧
        (跳过帧记入 submitted, 不回捞)。返回是否实际提交。"""
        if self.min_dist > 0 and pos is not None and np.isfinite(pos).all():
            near = self._last_pos is not None and \
                np.linalg.norm(pos - self._last_pos) < self.min_dist
            stale = self._last_idx is not None and \
                (kf_idx - self._last_idx) >= self.max_gap
            if near and not stale:
                self.submitted.add(kf_idx)
                return False
            self._last_pos = np.asarray(pos, np.float64)
        self._last_idx = kf_idx
        self.submit(kf_idx, frame_id)
        return True

    def catch_up(self, keyframes, pos_fn=None):
        """追赶式提交: 把 keyframes 容器里所有还没提交的关键帧补交
        (统一覆盖 INIT/TRACKING/backend重定位 三种 append 来源)。
        pos_fn(frame_id)->(3,) 米制位置(VIO), 供空间抽稀; None=不抽稀全提交。
        (不可用 SLAM T_WC 当位置源: Sim3 尺度自由且在线漂移, cfds_floor1 实测
        位姿量级从 ~10 漂到 1e20, 曾致前段过度抽稀 25%、中段 99% 全提交。)"""
        n = len(keyframes)
        for i in range(n):
            if i in self.submitted:
                continue
            with keyframes.lock:
                fid = int(keyframes.dataset_idx[i])
            self.submit_thinned(i, fid, pos_fn(fid) if pos_fn is not None else None)

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
        self._last_pos = None
        self._last_idx = None

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
        import io

        import requests
        from PIL import Image

        if not img_paths:
            return None
        content = []
        for p in img_paths:
            raw = p.read_bytes()
            if not p.stem.endswith("_0"):
                # 鱼眼缩半再发: 畸变大且不指望读文字, 1024x819 -> 512x410
                # 每张 ~835 -> ~250 视觉 token; 前视 _0 保持原图用于读门牌/公司名
                im = Image.open(io.BytesIO(raw))
                im = im.resize((im.width // 2, im.height // 2), Image.BILINEAR)
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "JPEG", quality=85)
                raw = buf.getvalue()
            b64 = base64.b64encode(raw).decode()
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


def _name_sim(x, y):
    """节点名相似度 [0,1]: 序列匹配比 与 字符二元组 Jaccard 取大。
    兼顾 OCR 变体('创享大厦'/'创意大厦')与词序差异('酒店入口'/'酒店出入口'),
    而语义不同的名('酒店大堂'/'前台接待区')得分低 —— 用于保守去重的语义门槛。"""
    if not x or not y:
        return 0.0
    seq = difflib.SequenceMatcher(None, x, y).ratio()
    ba = {x[i:i + 2] for i in range(len(x) - 1)} or {x}
    bb = {y[i:i + 2] for i in range(len(y) - 1)} or {y}
    jac = len(ba & bb) / len(ba | bb) if (ba | bb) else 0.0
    return max(seq, jac)


def aggregate_nodes(ann_by_kf, pos_by_kf, merge_dist=2.5, min_conf=0.5,
                    dedup_dist=2.0, name_sim_thresh=0.8):
    """把逐关键帧标注聚合成语义节点列表。

    ann_by_kf: {kf_idx: annotation dict}   (可直接传 Manager.dict 的快照)
    pos_by_kf: {kf_idx: (x, y, z) 世界系位置}  (调用方决定 VIO/SLAM 坐标)
    规则: 按 kf_idx 序把地标标注分段(同类别、相邻空间距离<merge_dist 且文字标识
          兼容才连成段 —— A电梯/B电梯这类编号不同的近邻地标不合并);
          段内代表优先取"读到文字标识"里置信度最高者(节点名尽量用真实门牌/公司名);
          再跨段做同类近距且标识兼容的合并(走回头路去重);
          最后一步「保守名字去重」把上游 signage 噪声(同一门口读出不同外卖柜/门牌字
          样、或同名 OCR 变体)拆出的重复节点按 名字相似度 合回 —— 同类 + 质心距 <
          dedup_dist + 名字相似 >= name_sim_thresh 才并, 语义不同的近邻(酒店大堂 vs
          前台)名字不像故保留, 新语义不丢。
    节点 position 取「离质心最近的成员关键帧位置」(medoid): 关键帧必在轨迹上,
          不会像质心那样因走回头路/回环漂移落进墙里/不可通行区。
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

    # 2.5) 保守名字去重: 同类 + medoid距 < dedup_dist + 名字相似 >= name_sim_thresh 的组合并。
    #      纯加合并(不拆已有组), 修 signage 噪声把同一地标拆成多个的情况。
    #      距离用 medoid(离质心最近的成员帧, 即最终显示位置)而非质心 —— 组内成员若因
    #      上游 signage 无限距合并横跨多处, 质心会落到中间空地, 使实际近在咫尺(实测 0.6m)
    #      的同名组被误判成十几米外不合并; medoid 是真实轨迹点, 与显示一致、不受横跨影响。
    def _rep_name(m):
        sc = [(bool(_sigs(a)), a.get("confidence", 0)) for a in m["anns"]]
        return m["anns"][int(max(range(len(sc)), key=lambda i: sc[i]))].get("name", "")

    def _medoid(m):
        P = np.asarray(m["positions"], np.float64)
        return P[int(np.argmin(np.linalg.norm(P - P.mean(axis=0), axis=1)))]
    meds = [_medoid(m) for m in merged]
    names = [_rep_name(m) for m in merged]
    parent = list(range(len(merged)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            if merged[i]["category"] != merged[j]["category"]:
                continue
            if np.linalg.norm(meds[i] - meds[j]) >= dedup_dist:
                continue
            if _name_sim(names[i], names[j]) < name_sim_thresh:
                continue
            parent[_find(i)] = _find(j)
    groups = {}
    for i in range(len(merged)):
        groups.setdefault(_find(i), []).append(i)
    deduped = []
    for idxs in groups.values():
        base = merged[idxs[0]]
        for k in idxs[1:]:
            base["kf_indices"] += merged[k]["kf_indices"]
            base["positions"] += merged[k]["positions"]
            base["anns"] += merged[k]["anns"]
            base["sigs"] |= merged[k]["sigs"]
        deduped.append(base)
    merged = deduped

    # 3) 每组选代表: 读到文字标识的优先 (节点名用真实门牌/公司名), 再比置信度。
    #    position 用 medoid(离质心最近的成员帧位置), rep_kf 仍用读字最清的帧(缩略图用)。
    nodes = []
    for m in merged:
        scores = [(bool(_sigs(a)), a.get("confidence", 0)) for a in m["anns"]]
        best = int(max(range(len(scores)), key=lambda i: scores[i]))
        rep = m["anns"][best]
        pos = np.asarray(m["positions"], np.float64)
        medoid = int(np.argmin(np.linalg.norm(pos - pos.mean(axis=0), axis=1)))
        nodes.append({
            "category": m["category"],
            "name": rep.get("name", ""),
            "description": rep.get("description", ""),
            "confidence": float(rep.get("confidence", 0)),
            "kf_indices": [int(k) for k in m["kf_indices"]],
            "rep_kf": int(m["kf_indices"][best]),
            "position": [float(x) for x in pos[medoid]],
        })
    return nodes

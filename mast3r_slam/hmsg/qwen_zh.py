"""HMSG 中文化 (L40 vLLM Qwen 接入构建过程):

1. 物体词表英->中批量翻译 (一次性, 缓存 vocab/scannet200_zh.json);
2. 房间命名: 每个房间把 物体构成 + 逐帧 Qwen 中文描述采样 + 门牌/公司名
   signage + 2 张代表帧图 交给 Qwen, 总结出最佳中文区域名/房型/一句话摘要
   (原版 generate_room_names 的 "label" 法升级: GPT->Qwen, 纯物体清单->多模态);
3. 中文查询翻译: CLIP 为英文模型, 查询前把中文翻成英文短语。

请求模式与 semantic.py 一致 (直连 vLLM, guided json, 绕过代理)。
"""
import base64
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

_VOCAB_ZH = pathlib.Path(__file__).parent / "vocab" / "scannet200_zh.json"

ROOM_NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 24},
        "room_type": {"type": "string", "maxLength": 12},
        "summary": {"type": "string", "maxLength": 80},
    },
    "required": ["name", "room_type", "summary"],
}

ROOM_NAME_PROMPT = """你是室内语义地图的区域命名员。下面是建图机器人对某个区域(房间/走廊)的全部观测信息, 请为它取一个**最好的中文区域名**。

【区域观测】
- 初步房型(CLIP 分类): {rtype}
- 区域内物体构成(top): {objects}
- 读到的文字标识(门牌/公司名等, 可能为空): {signage}
- 巡检时的逐帧描述采样:
{descs}
- 随附 {n_img} 张该区域代表帧画面

命名规则(重要):
1. 若文字标识里有**专属名称**(公司名/门牌号, 如"晟和新能源""2807"), 名字必须用它(可联合, 如"2807-PHYMI办公区");
2. 否则用"特征+功能"式简洁命名(如"落地窗开放办公区""电梯厅走廊""茶水休息区");
3. 名字 2-12 个字, 不带"左侧/前方"等方位词;
4. room_type 用简短中文房型词(办公区/走廊/会议室/打印区/茶水间/电梯厅/大厅/休息区/储物间/卫生间等);
5. summary 一句话概括该区域(有什么、什么特征)。

只输出 JSON: {{"name": "区域名", "room_type": "房型", "summary": "一句话描述"}}"""


def _chat(api_url, model, content, schema, timeout=90, max_tokens=300):
    import requests
    r = requests.post(
        f"{api_url.rstrip('/')}/chat/completions",
        json={"model": model,
              "messages": [{"role": "user", "content": content}],
              "max_tokens": max_tokens, "temperature": 0.0,
              "chat_template_kwargs": {"enable_thinking": False},
              "response_format": {"type": "json_schema",
                                  "json_schema": {"name": "out",
                                                  "schema": schema}}},
        timeout=timeout, proxies={"http": None, "https": None})
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


def translate_vocab(labels, api_url, model, batch=50):
    """英文词表 -> 中文 (批量, 结果缓存; 失败词保留英文)。返回 {en: zh}。"""
    cache = json.loads(_VOCAB_ZH.read_text()) if _VOCAB_ZH.exists() else {}
    todo = [x for x in labels if x not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        schema = {"type": "object",
                  "properties": {x: {"type": "string", "maxLength": 16}
                                 for x in chunk},
                  "required": list(chunk)}
        prompt = ("把下列室内物体类别名翻译成简体中文(常用叫法, 2-6字)。"
                  "只输出 JSON, 键为英文原词, 值为中文:\n" + ", ".join(chunk))
        try:
            out = _chat(api_url, model,
                        [{"type": "text", "text": prompt}], schema,
                        max_tokens=2000)
            cache.update({k: v.strip() for k, v in out.items() if v.strip()})
        except Exception as e:
            print(f"[hmsg-zh] 词表翻译批次失败(保留英文): {e}")
    _VOCAB_ZH.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return cache


def _b64(p, half=True):
    import io

    from PIL import Image
    im = Image.open(p)
    if half:
        im = im.resize((im.width // 2, im.height // 2), Image.BILINEAR)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def name_room(api_url, model, rtype, objects_zh, signage, descs, img_paths):
    """单房间命名: 观测汇总 + 代表帧图 -> {name, room_type, summary}。"""
    content = []
    for p in img_paths[:2]:
        if pathlib.Path(p).exists():
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{_b64(p)}"}})
    desc_txt = "\n".join(f"  - {d}" for d in descs[:6]) or "  (无)"
    content.append({"type": "text", "text": ROOM_NAME_PROMPT.format(
        rtype=rtype, objects=objects_zh or "(无)",
        signage=", ".join(signage) or "(无)", descs=desc_txt,
        n_img=min(2, len(img_paths)))})
    return _chat(api_url, model, content, ROOM_NAME_SCHEMA)


def localize_graph(g, dataset_dir, run_dir, seq, api_url, model, workers=8):
    """对 HMSGGraph 做中文化 (就地修改): 物体 name_zh + 房间 name_zh/type_zh/
    summary_zh。房间命名素材: 物体中文构成 + view 中文描述 + semantic.json 的
    signage/专属名 + 代表帧图。"""
    from collections import Counter

    # 1) 物体词表翻译
    en_labels = sorted({o.name for o in g.objects})
    zh = translate_vocab(en_labels, api_url, model)
    for o in g.objects:
        o.name_zh = zh.get(o.name, o.name)
    print(f"[hmsg-zh] 物体标签中文化: {len(en_labels)} 类")

    # 2) 帧号 -> signage/专属名 (semantic.json)
    fid_sig, fid_name = {}, {}
    semp = pathlib.Path(run_dir) / f"{seq}_semantic.json"
    if semp.exists():
        sem = json.loads(semp.read_text())
        fids = sem.get("frame_ids", [])
        for k, a in sem.get("annotations", {}).items():
            fid = a.get("frame_id", fids[int(k)] if int(k) < len(fids) else -1)
            if a.get("signage"):
                fid_sig[int(fid)] = a["signage"]
            if a.get("landmark") and a.get("name"):
                fid_name[int(fid)] = a["name"]

    ds = pathlib.Path(dataset_dir)
    views_by_room = {}
    for v in g.views:
        views_by_room.setdefault(v.room_id, []).append(v)

    def _one(r):
        objs = [o for o in g.objects if o.room_id == r.room_id]
        cnt = Counter(getattr(o, "name_zh", o.name) for o in objs)
        objects_zh = ", ".join(f"{n}x{c}" for n, c in cnt.most_common(12))
        vs = views_by_room.get(r.room_id, [])
        sigs, names, descs = set(), set(), []
        for v in vs:
            sigs |= set(fid_sig.get(v.img_id, []))
            if v.img_id in fid_name:
                names.add(fid_name[v.img_id])
            if v.vlm_description:
                descs.append(v.vlm_description)
        descs = descs[:: max(1, len(descs) // 6)]
        imgs = [ds / f"{fid:06d}.png" for fid in r.represent_images[:2]]
        try:
            out = name_room(api_url, model, r.name, objects_zh,
                            sorted(sigs | names)[:8], descs, imgs)
            r.name_zh = out["name"]
            r.type_zh = out["room_type"]
            r.summary_zh = out["summary"]
            print(f"[hmsg-zh] {r.room_id}: {r.name_zh} [{r.type_zh}] "
                  f"- {r.summary_zh}")
        except Exception as e:
            r.name_zh, r.type_zh, r.summary_zh = "", "", ""
            print(f"[hmsg-zh] 房间 {r.room_id} 命名失败: {e}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_one, g.rooms))
    return g


def translate_query(text, api_url, model):
    """中文查询 -> 英文短语 (CLIP 为英文模型)。ASCII 查询原样返回。"""
    if text.isascii():
        return text
    try:
        out = _chat(api_url, model, [{"type": "text", "text":
                    f"把这个室内物体/区域查询翻译成简短英文名词短语(2-5个词), "
                    f"只输出 JSON: {{\"en\": \"...\"}}\n查询: {text}"}],
                    {"type": "object",
                     "properties": {"en": {"type": "string", "maxLength": 48}},
                     "required": ["en"]}, timeout=30, max_tokens=60)
        return out["en"] or text
    except Exception:
        return text

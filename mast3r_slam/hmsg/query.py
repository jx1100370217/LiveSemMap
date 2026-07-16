"""HMSG fast 层级检索 (照抄 fsr_vln query_hierarchy_protected_icra 的三级
CLIP 匹配; 负词表用 icra 版 ["background"])。slow VLM 复核走 L40 Qwen
(detect_object_in_image 提示词照抄), 供导航/验收演示按需调用。"""
import numpy as np


class HMSGQuery:
    def __init__(self, graph, extractor):
        self.g = graph
        self.ext = extractor
        self._obj_feats = np.stack([o.embedding for o in graph.objects]) \
            if graph.objects else np.zeros((0, 768), np.float32)

    def query(self, text, top_k=5, negative=("background",)):
        """text -> (top 房间列表, top 物体列表)。
        房间: 查询与各房间代表视图特征取 max 相似, top5;
        物体: 候选=命中房间内物体, 负词表 argmax 过滤后按分数 top_k。"""
        tf = self.ext.encode_text([text, *negative])   # (1+neg, D)
        qf = tf[0]
        rooms = []
        for r in self.g.rooms:
            if not len(r.embeddings):
                continue
            s = float(np.max(np.stack(r.embeddings) @ qf))
            rooms.append((r, s))
        rooms.sort(key=lambda x: -x[1])
        top_rooms = rooms[:5]
        cand_ids = {r.room_id for r, _ in top_rooms}
        objs = []
        for o, f in zip(self.g.objects, self._obj_feats):
            if o.room_id not in cand_ids:
                continue
            sims = tf @ f                              # (1+neg,)
            if int(np.argmax(sims)) != 0:              # 最高分必须给 query 类
                continue
            objs.append((o, float(sims[0])))
        objs.sort(key=lambda x: -x[1])
        return top_rooms, objs[:top_k]

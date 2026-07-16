"""VLM 区域判定与命名的 prompt / schema。区域类型/形状不设枚举,
由 VLM 自由描述 (用户要求: 不限定死区域类型)。"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "same_space": {"type": "boolean"},
        "returned_region": {"type": "integer"},
        "space_kind": {"type": "string", "maxLength": 16},
        "space_summary": {"type": "string", "maxLength": 80},
        "shape": {"type": "string", "maxLength": 16},
        "my_position": {"type": "string", "maxLength": 48},
        "exit_reason": {"type": "string", "maxLength": 48},
    },
    "required": ["same_space", "returned_region", "space_kind",
                 "space_summary", "shape", "my_position", "exit_reason"],
}

JUDGE_PROMPT = """你是建图机器人的空间区域判定员。图1是机器人前视高清画面, 图2是同一时刻 4 路环视鱼眼拼图(格上标注了朝向: 前/右/后/左)。机器人在楼内持续移动, 你要判断它当前身处哪个空间区域。

【活动区域】(机器人此前所在的空间):
{cur_ctx}

【近期经过的历史区域】(id: 描述):
{hist_ctx}

请综合前视+4环视判断:
1. same_space: 当前所见是否**仍在活动区域那个物理空间内**?
   - 在同一空间内移动/转身/换位置, 只要没穿出这个空间的围合边界(墙/门/隔断划定), 都算 true;
   - 已穿过门/门洞/玻璃门, 或视觉可触及的空间结构明显换了(如从开阔办公区进入狭长走廊), 为 false。
2. returned_region: 若 same_space=false, 判断当前所见是否就是历史区域列表中的某一个(机器人走回了老地方, 画面特征与该区域描述吻合) -> **填该行 R 后面的数字编号**(列表已按空间距离排序, 距离近且画面吻合的优先); 不是任何历史区域(新地方)或 same_space=true -> 填 -1。
3. space_kind: 当前空间的类型, 用你认为最贴切的简短中文词组(不限词表, 如"开放办公区""货梯前室""景观阳台"...); shape: 平面形状的简短描述(如"长方形""L形""狭长""开阔不规则"...)。
4. space_summary: 当前空间的**身份描述**(60字内): 只写稳定特征——形状、围合方式(墙/玻璃隔断/落地窗方位)、功能(工位/会议桌/电梯门)、显著标识(公司名/门牌/绿植墙), 不写人员/光照等时变内容。
5. my_position: 机器人在该空间内的大致位置与朝向(如"位于区域东端, 面向西侧过道")。
6. exit_reason: same_space=false 时一句话说明变化(如"穿过玻璃门进入走廊"); true 时给空串。

判定原则: 空间身份看围合结构, 不看家具角度; 长走廊拐弯仍是同一走廊, 除非穿过了门; 开放办公区内部穿行不算离开; 门口/门槛上的帧算作即将进入的那个空间。
只输出 JSON。"""

NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 24},
        "room_type": {"type": "string", "maxLength": 12},
        "summary": {"type": "string", "maxLength": 90},
    },
    "required": ["name", "room_type", "summary"],
}

NAME_PROMPT = """机器人刚走完一个空间区域, 现在为它取名。信息如下:
- 区域内逐帧空间描述采样:
{summaries}
- 读到的文字标识(门牌/公司名, 可能为空): {signage}
- 随附 {n_img} 张该区域代表画面

命名规则:
1. 有专属名称(公司名/门牌号)必须用(可联合, 如"2807-PHYMI办公区");
2. 否则"特征+功能"式(如"落地窗开放办公区""电梯厅走廊");
3. 2-12 字, 不带方位词; room_type 用简短中文房型词; summary 一句话概括该区域的形状/围合/内容。
只输出 JSON: {{"name": "...", "room_type": "...", "summary": "..."}}"""

"""Versioned, uncalibrated image alert score and conditional offline guidance."""
import math

VERSION = 'image-alert-v1'
LEVELS = [
    (0, 24, '未检出或线索不足', '先复核'),
    (25, 49, '疑似烟火线索', '保持距离并核查'),
    (50, 74, '高优先级预警', '提醒人员并准备撤离'),
    (75, 100, '紧急优先级预警', '优先撤离并报警'),
]
FIRE_CONDITIONS = {
    'small_initial_fire': '现场确认为局限的初起小火',
    'trained': '操作人员受过灭火器培训',
    'clear_exit': '身后有畅通、安全的撤离路线',
    'no_dense_smoke': '无浓烟、无呼吸或视线受影响',
    'no_spread': '火势没有蔓延或迅速增大',
    'matching_extinguisher': '灭火器有效且适用于已确认的起火物',
    'no_special_hazard': '排除燃气泄漏、电池热失控和危险化学品',
    'alarm_arranged': '已经安排报警并提醒周围人员',
}
FIRE_TYPES = {'unknown', 'ordinary', 'cooking_oil', 'electrical', 'gas', 'battery'}


def validate_context(value):
    if not isinstance(value, dict) or set(value) - (set(FIRE_CONDITIONS) | {'fire_type'}):
        raise ValueError('现场确认条件格式无效')
    for key in FIRE_CONDITIONS:
        if key in value and not isinstance(value[key], bool):
            raise ValueError('现场确认条件必须为勾选值')
    if value.get('fire_type', 'unknown') not in FIRE_TYPES:
        raise ValueError('起火物类型无效')
    return value


def assess(record, context=None):
    context = context or {}
    state = record.get('vision_tools', {}).get('fire_smoke', {})
    available = state.get('status') == 'ok'
    if record.get('detection_source') == 'external_json_unverified' and not record.get('tool_error'):
        available = True
    size = (state.get('result') or record.get('vision_result') or {}).get('image_size')
    image_area = size[0] * size[1] if size and len(size) == 2 else None
    components = []
    max_confidence = {'fire': None, 'smoke': None}
    max_area = {'fire': 0.0, 'smoke': 0.0}
    candidates = []
    for detection in record.get('detections', []):
        label = detection.get('label')
        if label not in max_confidence:
            continue
        confidence = detection['confidence']
        max_confidence[label] = max(max_confidence[label] or 0, confidence)
        box = detection.get('box')
        area = 0.0
        if box and image_area:
            x1, y1, x2, y2 = box
            area = min(1.0, max(0.0, x2-x1) * max(0.0, y2-y1) / image_area)
        max_area[label] = max(max_area[label], area)
        value = (55 + 20*confidence + 15*math.sqrt(area) if label == 'fire'
                 else 25 + 25*confidence + 25*math.sqrt(area))
        candidates.append(value)
    both = all(max_confidence.values())
    score = min(100, round(max(candidates, default=0) + (10 if both else 0))) if available else None
    if score is None:
        level, label, priority = None, '未评分：检测未完成', '人工核查'
    else:
        level = next(i for i, (low, high, _, _) in enumerate(LEVELS) if low <= score <= high)
        _, _, label, priority = LEVELS[level]
    if candidates:
        components = [f"明火最高置信度：{max_confidence['fire']}",
                      f"烟雾最高置信度：{max_confidence['smoke']}",
                      '按每个目标的类别、置信度及框占图比例计算，取最高值；烟火并存加10分']
    elif available:
        components = ['没有达到模型检测阈值的烟火目标；0分表示没有检出线索，不代表现场安全']
    else:
        components = ['未执行或检测失败，不用0分代替未知结果']
    actions = {
        None: ['当前无法基于模型评分。不要靠近疑似烟火补拍；在安全位置人工核查。',
               '若现场可见火焰、浓烟或有人被困，直接优先撤离并报警，不等待模型结果。'],
        0: ['当前图片未提供明确检测线索，请在安全位置核查；可能存在漏检。',
            '若人工发现火焰、烟气、异味或报警器报警，按现场情况升级处置，不以0分解除警报。'],
        1: ['保持与疑似烟火的距离，提醒现场负责人和周围人员，核查实际烟气及安全出口。',
            '若确认发生火灾，应立即报警；有浓烟、蔓延或退路不明时撤离，不尝试扑救。'],
        2: ['提醒周围人员并向安全区域撤离；若确认发生火灾，立即报警。',
            '仅在初起小火和全部现场条件均确认满足时，受训人员才可考虑使用匹配的灭火器；任一条件不满足即撤离。'],
        3: ['优先远离烟火并沿安全路线撤离，提醒周围人员，不乘电梯、不返回火场。',
            '在安全位置拨打119，说明地址、起火物及被困情况；不要为取物或扑救延误撤离。'],
    }[level]
    checklist = {key: context.get(key) is True for key in FIRE_CONDITIONS}
    fire_type = context.get('fire_type', 'unknown')
    eligible = (level in {1, 2} and max_confidence['fire'] is not None and all(checklist.values()) and
                fire_type in {'ordinary', 'cooking_oil', 'electrical'})
    if eligible:
        extinguisher = ('你已勾选全部现场条件。仅在这些条件真实成立时，受训人员可考虑使用匹配的灭火器；'
                        '始终保留退路，火势增大、烟气加重或一次尝试未控制时立即撤离。')
    else:
        extinguisher = ('当前不能给出自行灭火许可：图片无法确认起火物、火势发展和退路；'
                        '需逐项现场确认全部条件，紧急预警时优先撤离。')
    methods = [
        '油锅起火：能安全操作时关闭热源，用合适锅盖或灭火毯覆盖；不要泼水，不要搬动燃烧油锅。',
        '带电设备：仅在安全条件下切断电源；未确认断电前不要用水，器材必须适用于带电设备。',
        '燃气泄漏、电池热失控或化学品：远离并撤离，交由专业救援人员处理，不按普通小火自行扑救。',
    ]
    return {'version': VERSION, 'score': score, 'level': level, 'label': label,
            'priority': priority, 'scope': 'uncalibrated_image_alert_not_fire_severity',
            'notice': '原型图片预警分：不是火灾概率、真实火势等级或自行灭火许可。未检出不能证明安全。',
            'max_confidence': max_confidence, 'max_box_area_ratio': max_area,
            'basis': components, 'actions': actions, 'extinguisher_eligible': eligible,
            'extinguisher_advice': extinguisher, 'field_checklist': checklist,
            'fire_type': fire_type, 'material_guidance': methods,
            'emergency_contact': '中国大陆火警119；网页不代拨。无公网不等于无蜂窝语音，能拨通时在安全位置报警；无法通话时用现场警报、对讲或其他可用方式求援。',
            'knowledge_status': 'project_curated_pending_local_review'}


def answer(assessment):
    score = f"{assessment['score']}/100" if assessment['score'] is not None else '未知'
    return (f"图片预警评分：{score}，{assessment['label']}。\n{assessment['notice']}\n"
            + '\n'.join(f'{i+1}. {action}' for i, action in enumerate(assessment['actions']))
            + '\n' + assessment['extinguisher_advice']
            + '\n' + assessment['emergency_contact'])


def risk_summary(assessment):
    """Use the image alert decision for the main risk heading as well."""
    level = assessment['level']
    if level is None:
        return ('无法评估：检测未完成',
                '本次烟火检测未执行或未成功完成，不能判定无风险；请在安全位置核查现场。')
    reason = f"图片预警评分 {assessment['score']}/100，{assessment['priority']}。"
    reason += {
        0: '本次未检出足够烟火线索，可能存在漏检，不能据此认定现场安全。',
        1: '发现疑似烟火线索，保持距离并核查；确认火灾后立即报警。',
        2: '已触发高优先级预警，应提醒人员并准备撤离；现场条件需进一步核实。',
        3: '已触发紧急预警，优先撤离并报警；起火物和现场条件仍需核实，不能等待核实后才避险。',
    }[level]
    return assessment['label'], reason

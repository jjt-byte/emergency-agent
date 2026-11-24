#!/usr/bin/env python3
"""Offline, human supervised incident record and report prototype."""
import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import tempfile
import uuid
from functools import lru_cache

import triage

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / "data"
DB = DATA / "knowledge.sqlite3"
REPORTS = DATA / "reports"
EVIDENCE = DATA / "evidence"
ALLOWED_LABELS = {"fire", "smoke", "person", "blocked_exit"}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=8)
def _model_digest(path, size, modified_ns):
    return sha256(path)


def model_digest(path):
    path = Path(path)
    stat = path.stat()
    return _model_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def connect():
    DATA.mkdir(exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL, body TEXT NOT NULL, added_at TEXT NOT NULL)")
    db.commit()
    return db


def add_doc(args):
    path = Path(args.file).expanduser().resolve()
    body = path.read_text(encoding="utf-8")
    if not body.strip():
        raise ValueError("知识文件为空")
    db = connect()
    cur = db.execute("INSERT INTO docs(title, source, body, added_at) VALUES (?, ?, ?, ?)",
                     (args.title or path.stem, str(path), body, utc_now()))
    db.commit()
    print(f"已导入条目 {cur.lastrowid}: {args.title or path.stem}，来源 {path}")


def retrieve(db, query, limit=3):
    words = [w for w in re.split(r"[\s，,。；;、：:！？!?]+", query) if len(w) >= 2]
    words += [w for w in ("明火", "烟雾", "人员", "通道", "疏散", "灭火器", "油锅", "带电", "报警", "撤离", "浓烟", "断网", "评分", "初起", "火灾", "燃气", "电池", "化学品") if w in query]
    if any(word in query for word in ('火灾', '着火', '起火', '火警', '119', '火情', '烟火', '预警', '火焰')):
        words += ['报警', '撤离']
    if any(word in query for word in ('逃生', '疏散', '被困')):
        words += ['撤离']
    words = list(dict.fromkeys(words))[:10]
    docs = db.execute("SELECT id,title,source,body FROM docs").fetchall()
    ranked = []
    for doc_id, title, source, body in docs:
        for index, raw_line in enumerate(body.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith(('#', '来源类型', '参考入口', '适用范围')):
                continue
            score = sum((3 if word in title else 0) + (1 if word in line else 0) for word in words)
            specific = [word for word in ('灭火器', '油锅', '带电', '燃气', '电池', '化学品', '断网', '评分') if word in query]
            score += sum((6 if word in title else 0) + (3 if word in line else 0) for word in specific)
            if score:
                curated = str(source).startswith(str(ROOT/'knowledge') + os.sep)
                ranked.append((score, {"doc_id": doc_id, "title": title, "source": source,
                                       "line": index, "excerpt": line[:350],
                                       "review_status": 'pending_local_review' if curated else 'user_provided_unverified'}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected, seen = [], set()
    for _, item in ranked:
        if item['doc_id'] in seen:
            continue
        seen.add(item['doc_id'])
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def parse_detections(path):
    if not path:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("检测文件必须是 JSON 数组")
    result = []
    for item in raw:
        if not isinstance(item, dict) or item.get("label") not in ALLOWED_LABELS:
            raise ValueError("检测项 label 必须是 fire/smoke/person/blocked_exit")
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence 必须是 0 到 1 的数字")
        parsed = {"label": item["label"], "confidence": float(confidence)}
        box = item.get("box")
        if box is not None:
            if (not isinstance(box, list) or len(box) != 4 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
                raise ValueError("box 必须是 [x1, y1, x2, y2] 数字数组")
            parsed["box"] = [float(v) for v in box]
        result.append(parsed)
    return result


def person_source():
    return {'damoyolo': 'damoyolo_person_cpu', 'cix_yolov8s': 'cix_yolov8s_npu'}.get(
        CONFIG.get('person_detection', {}).get('backend'), 'cix_ssd_300_npu')


def person_model_path(settings=None):
    settings = CONFIG.get('person_detection', {}) if settings is None else settings
    if settings.get('backend') == 'damoyolo':
        return Path(settings['model_dir'])/'damoyolo_tinynasL25_S_person.pt'
    if settings.get('backend') == 'cix_yolov8s':
        return Path(settings['model_path'])
    return Path(CONFIG['vision_model'])


def run_npu_detector(image_path):
    # Keep the original entry point for the SSD fallback and existing callers.
    settings = CONFIG.get('person_detection', {})
    if settings.get('backend') in {'damoyolo', 'cix_yolov8s'}:
        if settings['backend'] == 'cix_yolov8s':
            command = [settings['python'], str(ROOT/'tools/detect_yolov8_npu.py'), str(image_path),
                       '--model', settings['model_path'], '--threshold', str(settings.get('threshold', .5)),
                       '--nms-threshold', str(settings.get('nms_threshold', .45))]
        else:
            command = [settings['python'], str(ROOT/'tools/detect_person.py'), str(image_path),
                   '--model-dir', settings['model_dir'], '--threshold', str(settings.get('threshold', .5)),
                   '--max-side', str(settings.get('max_side', 640)), '--threads', str(settings.get('threads', 8))]
        result = subprocess.run(command, capture_output=True, text=True, timeout=240,
                                stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-1600:])
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict) or not isinstance(payload.get('detections'), list):
            raise ValueError('人体检测器返回格式错误')
        size = payload.get('image_size')
        if (not isinstance(size, list) or len(size) != 2 or
                any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)):
            raise ValueError('人体检测器未返回有效原图尺寸')
        if any(not isinstance(d, dict) or d.get('label') != 'person' for d in payload['detections']):
            raise ValueError('人体检测器出现不支持的类别')
        parsed = parse_detection_items(payload['detections'])
        if len(parsed) != len(payload['detections']) or any('box' not in d for d in parsed):
            raise ValueError('人体检测器返回无效置信度或坐标')
        return parsed, payload
    if not Path("/dev/aipu").exists():
        raise RuntimeError("当前运行环境找不到 /dev/aipu；请在板卡本机终端运行")
    detector = ROOT / "tools" / "detect_ssd.py"
    model = Path(CONFIG["vision_model"])
    command = [CONFIG.get("vision_python", sys.executable), str(detector), str(image_path), "--model", str(model)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-1200:])
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("detections"), list):
        raise ValueError("NPU 检测器返回格式错误")
    return parse_detection_items(payload["detections"]), payload


def run_fire_smoke_detector(image_path):
    settings = CONFIG.get('fire_smoke', {})
    runtime = Path(settings.get('python', '/nonexistent'))
    if not runtime.is_file():
        raise RuntimeError('烟火推理环境尚未安装，请运行 ./setup-fire-smoke.sh')
    command = [str(runtime), str(ROOT/'tools/detect_fire_smoke.py'), str(image_path),
               '--model-dir', settings['model_dir'], '--threshold', str(settings.get('threshold', .6)),
               '--max-side', str(settings.get('max_side', 640)), '--threads', str(settings.get('threads', 8))]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240,
                            stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-1600:])
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get('detections'), list):
        raise ValueError('烟火检测器返回格式错误')
    size = payload.get('image_size')
    if (not isinstance(size, list) or len(size) != 2 or
            any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)):
        raise ValueError('烟火检测器未返回有效原图尺寸')
    if any(not isinstance(d, dict) or d.get('label') not in {'fire', 'smoke'} for d in payload['detections']):
        raise ValueError('烟火检测器出现不支持的类别')
    parsed = parse_detection_items(payload['detections'])
    if len(parsed) != len(payload['detections']) or any('box' not in d for d in parsed):
        raise ValueError('烟火检测器返回无效置信度或坐标')
    return parsed, payload


def parse_detection_items(raw):
    result = []
    for item in raw:
        if not isinstance(item, dict) or item.get("label") not in ALLOWED_LABELS:
            continue
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            continue
        parsed = {"label": item["label"], "confidence": float(confidence)}
        if (isinstance(item.get("box"), list) and len(item["box"]) == 4 and
                all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                    for v in item['box'])):
            parsed["box"] = item["box"]
        result.append(parsed)
    return result


def classify(observation, detections, tool_error):
    if tool_error:
        return "待确认", "工具故障或输入失败，必须人工核查"
    strong_fire = any(d["label"] == "fire" and d["confidence"] >= CONFIG["fire_threshold"] for d in detections)
    strong_smoke = any(d["label"] == "smoke" and d["confidence"] >= CONFIG["smoke_threshold"] for d in detections)
    strong_block = any(d["label"] == "blocked_exit" and d["confidence"] >= 0.75 for d in detections)
    lower = observation.lower()
    negative_fire = any(w in lower for w in ("未发现明火", "没有发现明火", "未见明火", "无明火", "没有明火"))
    negative_smoke = any(w in lower for w in ("未发现烟雾", "没有发现烟雾", "未见烟雾", "无烟雾", "没有烟雾"))
    explicit_fire = not negative_fire and any(w in lower for w in ("发现明火", "看到明火", "有明火", "发现火焰", "看到火焰"))
    explicit_smoke = not negative_smoke and any(w in lower for w in ("发现烟雾", "看到烟雾", "有烟雾", "浓烟", "冒烟"))
    explicit_block = any(w in lower for w in ("通道堵塞", "出口堵塞", "通道被堵", "出口被堵"))
    if (strong_fire and negative_fire) or (strong_smoke and negative_smoke):
        return "待确认", "检测结果与人工观察冲突，必须重新核查"
    if strong_fire or explicit_fire:
        return "高风险待人工确认", "出现明火线索；立即提示负责人核查，处置由现场人员决定"
    if strong_smoke or strong_block or explicit_smoke or explicit_block:
        return "疑似风险待人工确认", "出现烟雾或通道受阻线索；建议现场复核并补充证据"
    if negative_fire or negative_smoke:
        return "未发现明确证据", "仅表示本次输入未提供明确火情证据，不能证明现场安全"
    return "待确认", "缺少足够的火情证据，需现场核查"


def request_plan(task, has_image=False, observation=''):
    """Choose evidence tools; all other questions remain open-ended local QA."""
    tutorial = r'怎么(?!样)|如何|怎样|原理|教程|方法|是什么|写.*代码'
    scene_reference = r'[这那](?:一)?(?:张|幅|个|些|里|种|边|是|儿)|图[片中里像]|画面|照片|现场|当前|现在'
    advice = r'(?:应该|应当|该|需要)?(?:怎么(?!样)|如何|怎样)(?:办|处理|应对|处置|疏散|撤离)|下一步|给.*建议|处置建议'
    asks_advice = bool(re.search(advice, task))
    counting = bool(re.search(r'几个人|几人|多少(?:个|名)?人|人数|人员数量|清点人员|数一[数下].*人|数数.*人|统计.*人', task))
    person = bool(re.search(r'人员|人群|行人|有人|这些人|他们|她们|人在哪|人的位置', task))
    risk = bool(re.search(r'风险|危险|安全|火情|烟雾|明火|火焰|火苗|烟火|火灾|浓烟|冒烟|有烟|通道|出口|有火|着火|起火|燃烧', task))
    visual = bool(re.search(r'图[片中里像]|画面|照片|识别|检测|检查|检察|看看|看一下|分析一下|穿.*[衣色]|颜色|受伤|倒地|位置|哪里|在哪', task))
    # A request to assess a scene and then suggest action still needs evidence.
    # Scope how-to markers to their clause, rather than vetoing the whole task.
    scene_request = False
    scene_advice = False
    for clause in re.split(r'[，,。；;！？!?]|并且|然后|同时|并', task):
        candidate = clause
        # "What risk is this?" with an attached scene asks for evidence.
        # Keep explicit requests for principles, tutorials and methods as QA.
        if (has_image and re.search(scene_reference, clause) and
                re.search(r'风险|危险|安全|隐患|火情|烟|火', clause) and
                not re.search(r'原理|教程|方法|写.*代码|解释|科普|如何|怎样|怎么(?!样|办)', clause)):
            candidate = candidate.replace('是什么', '')
        if re.search(r'几个人|几人|有多少|危险吗|安全吗|有没有|是否', clause):
            candidate = re.split(advice, candidate, maxsplit=1)[0]
        if re.search(scene_reference, clause) and re.search(advice, clause):
            candidate = re.sub(advice, '', candidate)
            if not re.search(r'人', candidate):
                scene_advice = not bool(re.search(tutorial, candidate)) or scene_advice
        if not re.search(tutorial, candidate) and re.search(
                r'人|风险|危险|安全|隐患|火|烟|通道|出口|检查|检测|检察', candidate):
            scene_request = True
    if re.search(r'原理|教程|方法|是什么|写.*代码|解释|科普', task) and not re.search(scene_reference, task):
        scene_request = False
    instructional = bool(re.search(tutorial, task)) and not scene_request and not scene_advice
    risk = risk or scene_advice
    unsupported = has_image and not instructional and bool(re.search(
        r'衣服|衣着|穿着|穿的|穿什么|颜色|受伤|伤情|倒地|摔倒|姿态|在干什么|在做什么|动作|身份|姓名|男女|性别|年龄', task))
    grounded = not instructional and (has_image or bool(observation) or visual or counting)
    kind = ('person_count' if grounded and counting else
            'visual_question' if unsupported else
            'person_inspection' if grounded and person else
            'risk_assessment' if grounded and risk else
            'visual_question' if has_image and visual and not instructional else 'general_question')
    return {'kind': kind, 'use_detection': has_image and kind in
            {'person_count', 'person_inspection', 'risk_assessment'},
            'show_people': kind in {'person_count', 'person_inspection'},
            'show_risk': kind == 'risk_assessment' or (grounded and risk),
            'unsupported_visual': unsupported, 'asks_advice': asks_advice}


def fire_smoke_summary(record):
    state = record.get('vision_tools', {}).get('fire_smoke', {})
    available = state.get('status') == 'ok' or (record.get('detection_source') == 'external_json_unverified'
                                               and not record.get('tool_error'))
    counts = {}
    for label in ('fire', 'smoke'):
        items = [d for d in record.get('detections', []) if d['label'] == label]
        counts[label] = {'count': len(items) if available else None,
                         'max_confidence': max((d['confidence'] for d in items), default=None)}
    return {'status': 'available' if available else 'unavailable', 'classes': counts,
            'threshold': state.get('result', {}).get('threshold'),
            'error': state.get('error'), 'source': state.get('source', record.get('detection_source', 'none'))}


def fire_smoke_answer(record, summary):
    if summary['status'] != 'available':
        if summary.get('error'):
            evidence = '烟火检测工具未成功完成，本次无法判断图片中的烟雾或明火，不能将失败当作未检出。'
        else:
            evidence = '本次未执行烟雾或明火检测；请提供图片并启用图片检测。'
    else:
        names = {'fire': '明火', 'smoke': '烟雾'}
        pieces = []
        for label, name in names.items():
            item = summary['classes'][label]
            pieces.append(f"{item['count']} 处{name}候选" if item['count'] else f'未检出{name}')
        evidence = '检测结果：' + '，'.join(pieces) + '。'
        if not any(item['count'] for item in summary['classes'].values()):
            evidence = '未检出达到检测阈值的烟火目标，不能据此证明现场安全。'
        if record.get('detection_source') == 'external_json_unverified':
            evidence += '本次为来源未验证的外部 JSON。'
    assessment = record.get('response_assessment')
    label = record.get('risk_level', '待确认')
    if not assessment:
        return evidence + '\n风险提示：' + label + '。'
    score = assessment['score']
    grade = label + (f'（图片预警分 {score}/100）' if score is not None else '')
    # User observations can raise the alert even when the image model missed it.
    level = next((i for i, item in enumerate(triage.LEVELS) if item[2] == label), assessment['level'])
    action = {
        None: '请在安全位置人工核查；现场可见烟火或有人被困时，立即撤离并拨打119。',
        0: '请在安全位置复核；现场发现烟火时立即撤离并拨打119。',
        1: '保持距离并核查，提醒周围人员；确认火灾后拨打119，有浓烟或蔓延时立即撤离。',
        2: '提醒周围人员并向安全区域撤离；确认火灾后在安全位置拨打119。',
        3: '立即远离烟火，提醒周围人员沿安全路线撤离，在安全位置拨打119；不要为扑救延误撤离。',
    }[level]
    if level in {1, 2}:
        action += ('仅在已确认的全部现场条件持续满足时，受训人员可考虑使用匹配的灭火器；火势增大即撤离。'
                   if assessment['extinguisher_eligible'] else '灭火条件未确认，请勿自行扑救。')
    if level != assessment['level']:
        grade += '；现场补充观察已触发升级，图片分数不降低预警'
    return evidence + '\n风险提示：' + grade + '。\n' + action


def with_task_result(record):
    """Attach evidence and task-specific defaults; preserve completed QA answers."""
    enriched = dict(record)
    enriched['vision_missing_capabilities'] = [label for label in record.get('vision_missing_capabilities', [])
                                              if label != 'blocked_exit']
    source = record.get('detection_source', 'none')
    person_state = record.get('vision_tools', {}).get('person')
    available = (person_state.get('status') == 'ok' if person_state is not None else
                 not record.get('tool_error') and source in {'cix_ssd_300_npu', 'cix_yolov8s_npu', 'damoyolo_person_cpu', 'external_json_unverified'})
    people = [d for d in record.get('detections', []) if d['label'] == 'person']
    review_threshold = 0.7
    high = sum(d['confidence'] >= review_threshold for d in people)
    statistics = {'status': 'available' if available else 'unavailable',
                  'estimated_count': len(people) if available else None,
                  'high_confidence_count': high if available else None,
                  'review_count': len(people) - high if available else None,
                  'review_threshold': review_threshold,
                  'detection_threshold': ((person_state or {}).get('result') or record.get('vision_result') or {}).get('threshold'),
                  'scope': 'single_image', 'basis': 'person_detections_after_detector_nms'}
    if available:
        if people:
            answer = f'本张图片估计有 {len(people)} 个人员目标。'
            if len(people) > high:
                answer += f'其中 {len(people)-high} 个检测框置信度较低，需复核。'
        else:
            answer = '本张图片未检出人员目标，不能据此认定现场无人。'
        if source == 'external_json_unverified':
            answer += '本次统计来自外部 JSON，来源尚未验证。'
    elif record.get('tool_error') or source.endswith('_failed'):
        answer = '人员检测工具未成功完成，当前无法统计人数；不能将失败记为 0 人。'
    else:
        answer = '本次未执行人员检测，当前无法统计人数。请启用人员检测后重新提交图片。'
    # A completed historical task retains the plan that actually ran its tools.
    plan = (record.get('request_plan') if record.get('answer_version') == 3 else None) or request_plan(
        record.get('task', ''), bool(record.get('image')), record.get('observation', ''))
    kind = plan['kind']
    smoke_summary = fire_smoke_summary(record)
    enriched.update(person_statistics=statistics, fire_smoke_statistics=smoke_summary,
                    task_kind=kind, request_plan=plan, answer_version=3)
    risk_updated = False
    if plan['show_risk'] and record.get('response_assessment'):
        level, reason = triage.risk_summary(record['response_assessment'])
        observed_level, observed_reason = classify(record.get('observation', ''), [], False)
        observed_floor = {'高风险待人工确认': 2, '疑似风险待人工确认': 1}.get(observed_level)
        image_level = record['response_assessment']['level']
        if observed_floor is not None and (image_level is None or observed_floor > image_level):
            level = triage.LEVELS[observed_floor][2]
            reason = ('现场补充观察触发升级预警（用户描述，未经独立核实）：' + observed_reason +
                      '。图片预警分仅反映模型输出，不能用于降低现场观察触发的预警。')
        risk_updated = (record.get('risk_summary_version') != 1 or
                        record.get('risk_level') != level or record.get('rule_reason') != reason)
        if risk_updated:
            # Preserve the previous wording without changing detection evidence.
            enriched.setdefault('previous_risk_summary', {
                'risk_level': record.get('risk_level'), 'rule_reason': record.get('rule_reason'),
                'task_answer': record.get('task_answer'),
            })
        enriched.update(risk_level=level, rule_reason=reason, risk_summary_version=1)
        record = enriched
    # Historical fixed-count answers must be recomputed once. New answers survive
    # report rendering, history reads and human confirmation.
    compact_needed = (kind in {'person_count', 'person_inspection', 'risk_assessment'}
                      and record.get('answer_presentation_version') != 1
                      and (plan['show_risk'] or record.get('answer_source') != 'local_qwen'))
    if not risk_updated and not compact_needed and record.get('task_answer') and (record.get('answer_version') == 3 or
                                     record.get('answer_version') == 2 and not plan['show_risk']):
        return enriched
    if kind in {'person_count', 'person_inspection'}:
        if kind == 'person_inspection' and available:
            answer = '人员检测结果：' + answer
            answer += '当前模型提供人员框和置信度，不能确认身份、伤情、姿态或衣着。'
        if plan['show_risk']:
            answer += '\n' + fire_smoke_answer(record, smoke_summary)
        if plan['unsupported_visual']:
            answer += '\n关于你询问的其他图像细节：当前人员检测工具无法识别衣着、伤情、姿态、身份等信息。'
    elif kind == 'risk_assessment':
        answer = fire_smoke_answer(record, smoke_summary)
        if any(word in record.get('task', '') for word in ('通道', '出口')):
            answer += '\n通道阻塞检测尚未接入，无法从图片确认通行情况。'
    elif kind == 'visual_question':
        answer = ('目前图片工具支持人员及烟火检测；尚未接入通用图像问答模型。'
                  '不能根据图片判断衣着、伤情、动作或其他未支持的内容。可补充文字描述后提问。')
    else:
        answer = '请启用“本地问答”后提交，由本机 Qwen 回答这个问题。'
    if compact_needed:
        if record.get('task_answer'):
            enriched.setdefault('previous_task_answer', record['task_answer'])
        if record.get('evidence_answer'):
            enriched.setdefault('previous_evidence_answer', record['evidence_answer'])
            enriched['evidence_answer'] = answer
        enriched['answer_presentation_version'] = 1
    enriched['task_answer'] = answer
    return enriched


def explain(record):
    model = Path(CONFIG["model_path"])
    cli = Path(CONFIG["llama_cli"])
    if not model.is_file() or not cli.is_file():
        return None, "本地模型或推理程序缺失"
    system = ('你是运行在本机的中文助手。直接回答用户当前的问题，遵循用户要求的内容和格式，简洁清楚。'
              '先给结论，再给必要的建议；默认用2到4句回答，不重复问题、不加套话、不重复已有结论。'
              '用户明确要求详细解释或指定格式时按其要求展开。'
              '普通问答不需要统计人数或进行火情分级。不能联网；不知道的事实明确说不知道。')
    context = ''
    if record['task_kind'] != 'general_question':
        system += ('你不能直接看图片，只能使用提供的工具证据。不得编造图像细节。'
                   '不能识别衣着、伤情、姿态或身份。人数以工具统计为准，风险以规则为准。'
                   f"本次成功检测的类别：{record.get('vision_capabilities', [])}；未成功执行的类别不得编造结果。")
        context += f"已知证据：{record['task_answer']}\n"
        if record['request_plan'].get('asks_advice'):
            system += ('用户同时询问怎么应对，请给出简短、具体的避险步骤。'
                       '区分检测线索与已确认事实；不能编造出口位置、起火原因或现场人员。'
                       '如现场确有烟火，建议远离烟火并撤离、提醒周围人员，通信可用时拨打119，'
                       '不要返回火场或乘电梯。断网时不能承诺网页代拨电话。'
                       '如果检测没有发现目标，应明确无法排除漏检，建议安全距离人工核查。')
        if (record.get('vision_result') or {}).get('image_size'):
            context += f"图片尺寸：{record['vision_result']['image_size']}\n"
        boxes = [{'编号': i + 1, '类别': d['label'], '坐标': [round(v, 1) for v in d.get('box', [])],
                  '置信度': round(d['confidence'], 2)} for i, d in enumerate(record['detections'][:8])]
        context += f"检测框坐标（左上到右下）：{boxes}\n"
    elif record.get('image'):
        system += '用户附有图片，但你不能直接看图；不能臆测图片内容。'
    if record.get('observation'):
        context += f"用户补充描述（未经现场核实）：{record['observation'][:600]}\n"
    if record.get('knowledge_refs'):
        system += '本地资料优先；项目整理且待审核的内容不能说成官方结论，未知现场条件不得猜测。'
        context += f"本地资料参考：{[r['excerpt'][:120] for r in record['knowledge_refs'][:3]]}\n"
    # Escape chat delimiters in user-controlled text so they remain plain data.
    clean = lambda value: value.replace('<|im_start|>', '').replace('<|im_end|>', '')
    prompt = ("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" +
              clean(context + '\n用户问题：' + record['task']) + "<|im_end|>\n<|im_start|>assistant\n")
    cmd = ["taskset", "-c", CONFIG["cpu_affinity"], str(cli), "-m", str(model),
           "-p", prompt, "-t", str(CONFIG["threads"]), "-n", "384", "-c", "4096",
           "--temp", "0.2",
           "--no-conversation", "--no-display-prompt", "--simple-io", "--no-warmup"]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=150, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if run.returncode:
        return None, (run.stderr or run.stdout)[-500:]
    answer = run.stdout.replace("[end of text]", "").split('<|im_end|>')[0].strip()
    return (answer[:3000] if answer else None), (None if answer else "模型未生成文本")


def run_case(args, fire_detector=None):
    if getattr(args, 'agent', False) and args.llm and not args.detections:
        from agent_loop import run_agent
        DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='agent-cache-', dir=DATA) as cache:
            args.agent_prompt_cache = str(Path(cache)/'prompt.bin')
            return run_agent(args)
    started = time.perf_counter()
    db = connect()
    detections = []
    detection_source = "none"
    vision_result = None
    tool_error = args.tool_error
    tool_error_message = None
    events = [{"step": "plan", "status": "ok", "at": utc_now()}]
    plan = request_plan(args.task, bool(args.image), args.observation or '')
    use_detection = args.npu_detect and plan['use_detection']
    vision_tools = {'person': {'status': 'skipped', 'source': person_source()},
                    'fire_smoke': {'status': 'skipped', 'source': 'damoyolo_cpu'}}
    print(f"本次任务：{plan['kind']}；按问题选择证据工具并生成回答", flush=True)
    if use_detection:
        image_path = Path(args.image).expanduser().resolve()
        wants_person = plan['show_people'] or not plan['show_risk']
        wants_fire = plan['show_risk'] and CONFIG.get('fire_smoke', {}).get('enabled', False)
        for name, requested, detector in [('person', wants_person, run_npu_detector),
                                           ('fire_smoke', wants_fire, fire_detector or run_fire_smoke_detector)]:
            if not requested:
                continue
            state = vision_tools[name]
            try:
                items, payload = detector(image_path)
                if (vision_result and vision_result.get('image_size') and payload.get('image_size') and
                        vision_result['image_size'] != payload['image_size']):
                    raise ValueError('不同检测工具返回的原图尺寸不一致')
                detections.extend(items)
                state.update(status='ok', result=payload)
                if not vision_result:
                    vision_result = dict(payload)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                state.update(status='failed', error=str(exc)[-1600:])
                tool_error = True
                print(f"{name} 检测失败：{exc}", file=sys.stderr)
            events.append({'step': name, 'status': state['status'], 'at': utc_now()})
        sources = [state['source'] for state in vision_tools.values() if state['status'] == 'ok']
        detection_source = '+'.join(sources) or '+'.join(
            state['source']+'_failed' for state in vision_tools.values() if state['status'] == 'failed') or 'none'
        errors = [f"{name}: {state['error']}" for name, state in vision_tools.items() if state.get('error')]
        tool_error_message = '\n'.join(errors) or None
        if vision_result:
            vision_result['detections'] = detections
    elif args.detections:
        detections = parse_detections(args.detections)
        detection_source = "external_json_unverified"
        # The legacy external JSON interface cannot attest tool execution.
        vision_tools = {}
    events.append({"step": "vision", "status": "failed" if tool_error else
                   ("ok" if detection_source != "none" else "skipped"), "at": utc_now()})
    case_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    REPORTS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    image = None
    if args.image:
        source = Path(args.image).expanduser().resolve()
        if not source.is_file():
            raise ValueError("图片不存在")
        target = EVIDENCE / f"{case_id}{source.suffix.lower()}"
        shutil.copy2(source, target)
        image = {"path": str(target), "sha256": sha256(target), "size_bytes": target.stat().st_size}
    risk_error = args.tool_error or vision_tools.get('fire_smoke', {}).get('status') == 'failed'
    level, reason = classify(args.observation or "", detections, risk_error)
    knowledge_query = f"{args.task} {args.observation or ''}"
    if plan['show_risk']:
        knowledge_query += ' 烟雾 明火 报警 撤离 灭火器'
    refs = retrieve(db, knowledge_query)
    db.close()
    events.extend([{"step": "retrieve", "status": "ok" if refs else "no_match", "at": utc_now()},
                   {"step": "classify", "status": "ok", "at": utc_now()}])
    if getattr(args, "confirm_after", False):
        print(f"检测结果：{detections or '未检出人员或未提供检测'}\n风险提示：{level}\n规则依据：{reason}")
        try:
            args.confirm = input("人工复核记录（未确认直接回车）：").strip() or None
        except (EOFError, KeyboardInterrupt):
            print("\n未收到人工确认，保存为未确认。")
            args.confirm = None
    vision_model = person_model_path()
    capabilities = []
    if vision_tools.get('person', {}).get('status') == 'ok':
        capabilities.append('person')
    if vision_tools.get('fire_smoke', {}).get('status') == 'ok':
        capabilities += ['fire', 'smoke']
    if detection_source == 'external_json_unverified':
        capabilities = sorted({d['label'] for d in detections})
    record = {
        "case_id": case_id, "created_at": utc_now(), "task": args.task,
        "observation": args.observation or "", "detections": detections,
        "detection_source": detection_source,
        "vision_result": vision_result,
        "vision_tools": vision_tools,
        "vision_model_path": str(vision_model) if vision_tools.get('person', {}).get('status') in {'ok', 'failed'} else None,
        "vision_model_sha256": model_digest(vision_model) if 'person' in capabilities and vision_model.is_file() else None,
        "fire_smoke_model_path": CONFIG.get('fire_smoke', {}).get('model_dir') if plan['show_risk'] and use_detection else None,
        "vision_capabilities": capabilities,
        "vision_missing_capabilities": sorted({'person', 'fire', 'smoke'}-set(capabilities)) if args.image else [],
        "tool_error": tool_error, "tool_error_message": tool_error_message,
        "events": events,
        "image": image, "risk_level": level, "rule_reason": reason,
        "knowledge_refs": refs, "human_confirmation": args.confirm or "未确认",
        "model_path": str(Path(CONFIG["model_path"])),
        "model_sha256": model_digest(CONFIG["model_path"]) if Path(CONFIG["model_path"]).is_file() else None,
        "llm_explanation": None, "llm_error": None,
    }
    if getattr(args, 'batch_id', None):
        record['dataset_sample'] = {'batch_id': args.batch_id, 'name': Path(args.image).name}
    if plan['show_risk']:
        record['safety_context'] = getattr(args, 'safety_context', None) or {}
        record['response_assessment'] = triage.assess(record, record['safety_context'])
    record = with_task_result(record)
    record['evidence_answer'] = record['task_answer'] if plan['kind'] != 'general_question' else None
    if plan['show_people']:
        events.append({"step": "count_people", "status": record['person_statistics']['status'], "at": utc_now()})
    # Keep safety advice in versioned local rules rather than a small model's draft.
    generate_answer = args.llm and plan['kind'] != 'visual_question' and not plan['show_risk']
    if generate_answer:
        print("正在调用本地 Qwen 回答当前问题……", flush=True)
        record["llm_explanation"], record["llm_error"] = explain(record)
        if plan['kind'] in {'general_question', 'person_inspection'}:
            record['task_answer'] = record['llm_explanation'] or (
                record['evidence_answer'] or '本地问答未完成，请重试。')
            record['answer_source'] = 'local_qwen' if record['llm_explanation'] else 'llm_failed'
        else:
            record['answer_source'] = 'local_evidence'
    else:
        record['answer_source'] = 'local_evidence' if plan['kind'] != 'general_question' else 'llm_disabled'
    events.append({"step": "explain", "status": "failed" if record["llm_error"] else
                   ("ok" if generate_answer else "skipped"), "at": utc_now()})
    events.append({"step": "human_confirmation", "status": "recorded" if args.confirm else "pending", "at": utc_now()})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    events.append({"step": "save_report", "status": "ok", "at": utc_now()})
    json_path, md_path, html_path = save_record(record)
    print(f"任务回答：{record['task_answer']}\n风险提示：{record['risk_level']}\n规则依据：{record['rule_reason']}\n人工确认：{record['human_confirmation']}")
    print(f"报告：{md_path}\n原始记录：{json_path}")
    print(f"图片和检测框：{html_path}")
    if record["llm_error"]:
        print(f"LLM 说明未生成：{record['llm_error']}", file=sys.stderr)
    return record


def save_record(record):
    """Render all report formats from the same saved record."""
    record = with_task_result(record)
    case_id = record["case_id"]
    detections = record["detections"]
    detection_source = record["detection_source"]
    vision_result = record.get("vision_result")
    tool_error_message = record.get("tool_error_message")
    image = record["image"]
    refs = record["knowledge_refs"]
    level, reason = record["risk_level"], record["rule_reason"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / f"{case_id}.json"
    md_path = REPORTS / f"{case_id}.md"
    temporary_json = json_path.with_suffix('.tmp')
    temporary_json.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_json.replace(json_path)
    refs_md = "\n".join(f"- {r['title']}，第 {r['line']} 行：{r['excerpt']}（{r['source']}）" for r in refs) or "- 未检索到匹配的本地条目"
    agent_md = ('Agent执行记录：\n' + '\n'.join(
        f"- {step['title']}：{step['status']}；{step.get('detail', '')}（{step.get('source', 'runtime')}）"
        for step in record.get('agent_trace', [])) + '\n\n' if record.get('agent_trace') else '')
    people_md = f"人员统计：{record['person_statistics']}\n\n" if record['request_plan']['show_people'] else ""
    risk_md = (f"风险提示：**{level}**\n\n规则依据：{reason}\n\n"
               if record['request_plan']['show_risk'] else "")
    supplement = record.get('llm_explanation')
    supplement_md = (f"本地 Qwen 补充回答（待复核）：\n\n{supplement}\n\n"
                     if supplement and supplement != record['task_answer'] else '')
    assessment_md = ''
    if record.get('response_assessment'):
        assessment = record['response_assessment']
        context = record.get('safety_context') or {}
        conditions = '\n'.join(f"- {description}：{'用户已勾选确认' if context.get(key) is True else '未确认/不满足'}"
                               for key, description in triage.FIRE_CONDITIONS.items())
        assessment_md = ('完整处置参考：\n' + triage.answer(assessment) + '\n\n'
                         + '图片评分依据：\n' + '\n'.join('- '+line for line in assessment['basis']) + '\n\n'
                         + '现场条件（由用户填写，模型无法代替核实）：\n' + conditions + '\n\n'
                         + '按起火物确认后的参考（不能由图片确定材料）：\n'
                         + '\n'.join('- '+line for line in assessment['material_guidance']) + '\n\n'
                         + '知识来源状态：项目通用提示整理，待现场审核；机构原文尚未联网核验。\n\n')
    md = (f"# 离线巡查记录 {case_id}\n\n时间：{record['created_at']}\n\n任务：{record['task']}\n\n"
          f"任务回答：{record['task_answer']}\n\n{agent_md}{people_md}"
          f"观察：{record['observation'] or '未提供'}\n\n检测结果：{detections if detection_source != 'none' else '未接入'}\n\n"
          f"视觉来源：{detection_source}\n\n视觉能力：{record.get('vision_capabilities', [])}；未覆盖：{record.get('vision_missing_capabilities', [])}\n\n"
          f"工具错误：{tool_error_message or '无'}\n\n"
          f"{risk_md}{assessment_md}图片证据：{image['path'] if image else '未提供'}\n\n"
          f"本地依据：\n{refs_md}\n\n{supplement_md}"
          f"人工确认：{record['human_confirmation']}\n\n本报告仅供现场人员辅助研判。\n")
    md_path.write_text(md, encoding="utf-8")
    html_path = REPORTS / f"{case_id}.html"
    preview = ""
    if image:
        relative = "../evidence/" + Path(image["path"]).name
        size = (vision_result or {}).get("image_size")
        if size:
            width, height = size
            boxes = []
            person_number = 0
            for item in detections:
                if item['label'] == 'person':
                    person_number += 1
                if "box" not in item:
                    continue
                x1, y1, x2, y2 = item["box"]
                number = f' #{person_number}' if item['label'] == 'person' else ''
                label = html.escape(f"{item['label']}{number} {item['confidence']:.2f}")
                color = ('#dc4f32' if item['label'] == 'fire' else '#b38a19' if item['label'] == 'smoke'
                         else '#e5a234' if item['confidence'] < 0.7 else '#18e98c')
                boxes.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="3"/>'
                             f'<text x="{x1}" y="{max(18, y1-5)}" fill="{color}" font-size="18">{label}</text>')
            preview = (f'<svg viewBox="0 0 {width} {height}" style="max-width:960px;width:100%">'
                       f'<image href="{relative}" width="{width}" height="{height}"/>{"".join(boxes)}</svg>')
        else:
            preview = f'<img src="{relative}" style="max-width:960px;width:100%" alt="现场证据">'
    html_path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>离线巡查记录</title>'
                        '<body style="font-family:sans-serif;max-width:1100px;margin:24px auto;padding:16px">'
                        + preview + '<pre style="white-space:pre-wrap;line-height:1.7">' + html.escape(md)
                        + '</pre></body></html>', encoding="utf-8")
    return json_path, md_path, html_path


def interactive():
    print("断网应急巡查智能体。输入 /exit 退出；模型、知识库和报告均保存在本机。")
    print("图片任务按需调用配置中的人员或烟火检测模型；当前 DAMO-YOLO 使用本地 CPU。")
    while True:
        try:
            task = input("\n巡查任务：").strip()
            if task in {"/exit", "退出"}:
                return
            if not task:
                continue
            image = input("图片路径（可拖入终端，无图片直接回车）：").strip().strip("\"'")
            observation = input("现场观察（可留空）：").strip()
            args = argparse.Namespace(task=task, image=image or None, observation=observation,
                                      npu_detect=bool(image), detections=None, tool_error=False,
                                      confirm=None, confirm_after=True, llm=True)
            try:
                run_case(args)
            except (OSError, ValueError) as exc:
                print(f"任务失败：{exc}")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return


def main():
    parser = argparse.ArgumentParser(description="断网应急巡查辅助智能体原型")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="初始化本地目录和知识库")
    sub.add_parser("interactive", help="连续输入任务，自动调用本地检测和 GGUF")
    add = sub.add_parser("add-knowledge", help="导入 UTF-8 文本知识条目")
    add.add_argument("file")
    add.add_argument("--title")
    task = sub.add_parser("run", help="执行一条离线巡查任务")
    task.add_argument("--result-file", help=argparse.SUPPRESS)
    task.add_argument("--agent", action="store_true", help="由本地Qwen规划、调用工具并根据结果继续处理")
    task.add_argument("--agent-context", help=argparse.SUPPRESS)
    task.add_argument("--progress-file", help=argparse.SUPPRESS)
    task.add_argument("--safety-context", type=json.loads, default={})
    task.add_argument("--task", required=True)
    task.add_argument("--observation", default="")
    task.add_argument("--image")
    task.add_argument("--detections", help="外部视觉工具的检测结果 JSON 文件")
    task.add_argument("--npu-detect", "--detect", dest="npu_detect", action="store_true",
                      help="按任务执行配置中的本地人员或烟火图片检测")
    task.add_argument("--tool-error", action="store_true")
    task.add_argument("--confirm", help="现场人员确认文字")
    task.add_argument("--confirm-after", action="store_true", help="检测完成后询问人工复核结果")
    task.add_argument("--llm", action="store_true", help="调用本地 GGUF 生成辅助说明")
    args = parser.parse_args()
    try:
        if args.command == "init":
            connect().close()
            REPORTS.mkdir(parents=True, exist_ok=True)
            EVIDENCE.mkdir(parents=True, exist_ok=True)
            print(f"已初始化：{DATA}")
        elif args.command == "add-knowledge":
            add_doc(args)
        elif args.command == "interactive":
            interactive()
        else:
            if args.agent_context:
                args.agent_context = json.loads(Path(args.agent_context).read_text(encoding='utf-8'))
            if args.npu_detect and args.detections:
                raise ValueError("--npu-detect 与 --detections 请选择一种")
            record = run_case(args)
            if args.result_file:
                Path(args.result_file).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()

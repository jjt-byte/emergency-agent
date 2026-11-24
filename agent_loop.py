"""Bounded, offline Qwen tool loop. Tool evidence remains authoritative."""
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid
import datetime as dt

import agent
import triage

VERSION = 'local-tool-agent-v1'
TOOL_NAMES = {'detect_people', 'detect_fire_smoke', 'search_knowledge', 'assess_risk'}
TITLES = {'detect_people': '人员检测', 'detect_fire_smoke': '烟火检测',
          'search_knowledge': '本地知识检索', 'assess_risk': '预警评估'}
_BASE_SCHEMA = {
    'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['tools', 'finish', 'ask']},
        'tools': {'type': 'array', 'maxItems': 4, 'items': {
            'type': 'object', 'properties': {
                'name': {'type': 'string', 'enum': sorted(TOOL_NAMES)},
                'query': {'type': 'string'}}, 'required': ['name'], 'additionalProperties': False}},
        'answer': {'type': 'string', 'minLength': 1, 'maxLength': 500},
        'question': {'type': 'string', 'minLength': 3, 'maxLength': 120}},
    'required': ['action'], 'additionalProperties': False}


def action_schema(action):
    schema = copy.deepcopy(_BASE_SCHEMA)
    schema['properties']['action'] = {'const': action}
    if action == 'tools':
        schema['required'].append('tools')
        schema['properties']['tools']['minItems'] = 1
        schema['properties'].pop('answer')
        schema['properties'].pop('question')
    else:
        schema['properties'].pop('tools')
        schema['required'].append('question' if action == 'ask' else 'answer')
    return schema


SCHEMA = {'oneOf': [action_schema(action) for action in ['tools', 'finish']]}
FINAL_SCHEMA = action_schema('finish')


def complete(system, payload, max_tokens=300, schema=None, prompt_cache=None):
    """Use the installed llama.cpp CLI; no network or additional packages."""
    clean = lambda text: text.replace('<|im_start|>', '').replace('<|im_end|>', '')
    prompt = ('<|im_start|>system\n' + clean(system) + '<|im_end|>\n<|im_start|>user\n' +
              clean(json.dumps(payload, ensure_ascii=False, separators=(',', ':'))) +
              '<|im_end|>\n<|im_start|>assistant\n')
    config = agent.CONFIG
    command = ['taskset', '-c', config['cpu_affinity'], config['llama_cli'],
               '-m', config['model_path'], '-t', str(config['threads']), '-c', '4096',
               '-n', str(max_tokens), '--temp', '0.1', '--seed', '42', '--no-conversation',
               '--no-display-prompt', '--simple-io', '--no-warmup',
               '--json-schema', json.dumps(schema or SCHEMA), '-p', prompt]
    if prompt_cache:
        command.extend(['--prompt-cache', str(prompt_cache)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=140,
                            stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-600:])
    raw = result.stdout.replace('[end of text]', '').split('<|im_end|>')[0].strip()
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('本地模型未返回完整工具指令') from exc
    validate_decision(decision)
    return decision


def validate_decision(value):
    if not isinstance(value, dict) or value.get('action') not in {'tools', 'finish', 'ask'}:
        raise ValueError('无效的Agent动作')
    if set(value) - {'action', 'tools', 'answer', 'question'}:
        raise ValueError('Agent指令包含未支持的字段')
    for field in ['answer', 'question']:
        if field in value and (not isinstance(value[field], str) or len(value[field]) > 1800):
            raise ValueError('Agent回答格式无效')
    tools = value.get('tools', [])
    if not isinstance(tools, list) or len(tools) > 4:
        raise ValueError('工具列表无效')
    for item in tools:
        if (not isinstance(item, dict) or set(item) - {'name', 'query'} or
                item.get('name') not in TOOL_NAMES or
                not isinstance(item.get('query', ''), str) or len(item.get('query', '')) > 300):
            raise ValueError('工具名称或参数不在允许范围内')
    if value['action'] == 'tools' and not tools:
        raise ValueError('工具调用列表为空')
    if value['action'] != 'tools' and tools:
        raise ValueError('结束或追问动作不能同时调用工具')
    if value['action'] == 'ask' and not value.get('question', '').strip():
        raise ValueError('追问内容为空')
    if value['action'] == 'finish' and not value.get('answer', '').strip():
        raise ValueError('回答内容为空')
    return value


def fingerprint(tool):
    config_key = 'person_detection' if tool == 'detect_people' else 'fire_smoke'
    settings = agent.CONFIG.get(config_key, {})
    paths = [agent.ROOT/'tools/detect_fire_smoke.py']
    if tool == 'detect_people':
        paths.append(agent.ROOT/'tools/detect_person.py')
        paths.append(agent.ROOT/'tools/detect_yolov8_npu.py')
        paths.append(agent.person_model_path())
    if settings.get('model_dir'):
        paths.extend(Path(settings['model_dir']).glob('*.pt'))
    elif tool == 'detect_people':
        paths.append(Path(agent.CONFIG['vision_model']))
    return {'settings': settings, 'files': {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                                           for p in paths if p.exists()}}


def base_record(args):
    case_id = dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    image = None
    if args.image:
        source = Path(args.image).expanduser().resolve()
        if not source.is_file():
            raise ValueError('现场图片不存在')
        agent.EVIDENCE.mkdir(parents=True, exist_ok=True)
        target = agent.EVIDENCE/(case_id + source.suffix.lower())
        shutil.copy2(source, target)
        image = {'path': str(target), 'sha256': agent.sha256(target), 'size_bytes': target.stat().st_size}
    person = agent.CONFIG.get('person_detection', {})
    person_source = agent.person_source()
    observation = args.observation or ''
    if re.search(r'我(?:看到|发现)|现场.*(?:有明火|冒烟|浓烟)|现在.*着火', args.task):
        observation += '\n用户本轮现场描述：' + args.task
    return {'case_id': case_id, 'created_at': agent.utc_now(), 'task': args.task,
            'observation': observation.strip(), 'image': image, 'detections': [],
            'detection_source': 'none', 'vision_result': None,
            'vision_tools': {'person': {'status': 'skipped', 'source': person_source},
                             'fire_smoke': {'status': 'skipped', 'source': 'damoyolo_cpu'}},
            'vision_capabilities': [], 'vision_missing_capabilities': [],
            'tool_error': False, 'tool_error_message': None,
            'risk_level': '未评估', 'rule_reason': '本轮尚未执行预警评估',
            'knowledge_refs': [], 'human_confirmation': args.confirm or '未确认',
            'model_path': agent.CONFIG['model_path'], 'model_sha256': agent.model_digest(agent.CONFIG['model_path']),
            'vision_model_path': None, 'vision_model_sha256': None, 'fire_smoke_model_path': None,
            'llm_explanation': None, 'llm_error': None,
            'safety_context': getattr(args, 'safety_context', None) or {},
            'events': [], 'agent_trace': [], 'agent_tool_fingerprints': {},
            'agent_execution': {'version': VERSION, 'status': 'running', 'decisions': []}}


def refresh_evidence(record):
    """Build the existing report/UI schema from actual tool executions."""
    tools = record['vision_tools']
    record['detections'] = [d for state in tools.values() if state['status'] == 'ok'
                            for d in state['result']['detections']]
    record['vision_capabilities'] = ((['person'] if tools['person']['status'] == 'ok' else []) +
                                    (['fire', 'smoke'] if tools['fire_smoke']['status'] == 'ok' else []))
    record['vision_missing_capabilities'] = sorted({'person', 'fire', 'smoke'} - set(record['vision_capabilities'])) if record['image'] else []
    record['detection_source'] = '+'.join(state['source'] for state in tools.values() if state['status'] == 'ok') or 'none'
    errors = [f"{name}: {state['error']}" for name, state in tools.items() if state.get('error')]
    record['tool_error'] = bool(errors)
    record['tool_error_message'] = '\n'.join(errors) or None
    first = next((state['result'] for state in tools.values() if state['status'] == 'ok'), None)
    record['vision_result'] = dict(first, detections=record['detections']) if first else None
    return record


def planner_state(record, context, rounds_left):
    tools = {}
    for name, state in record['vision_tools'].items():
        result = state.get('result', {})
        counts = {}
        for detection in result.get('detections', []):
            counts[detection['label']] = counts.get(detection['label'], 0) + 1
        tools[name] = {'status': state['status'], 'counts': counts}
        if re.search(r'位置|哪里|在哪|左|右', record['task']):
            tools[name]['sample_boxes'] = [{**d, 'box': [round(v) for v in d.get('box', [])]}
                                           for d in result.get('detections', [])[:4]]
        if state.get('error'):
            tools[name]['error'] = state['error'][-200:]
    assessment = record.get('response_assessment')
    memory = {}
    if record['image'] and record['image']['sha256'] == context.get('image_sha256'):
        for name, item in context.get('cached_tools', {}).items():
            counts = {}
            for detection in item.get('state', {}).get('result', {}).get('detections', []):
                counts[detection['label']] = counts.get(detection['label'], 0) + 1
            memory[name] = {'counts': counts, 'case_id': item['case_id']}
    return {'question': record['task'], 'observation': record['observation'][-600:],
            'image_available': bool(record['image']), 'detection_allowed': context.get('detection_allowed', True),
            'history': [{'question': turn['question'][:80], 'answer': turn['answer'][:120]}
                        for turn in context.get('history', [])[-2:]], 'tools': tools,
            'risk': ({'score': assessment['score'], 'label': record['risk_level'],
                      'mandatory_guidance': ('先撤离并在安全位置拨打119，不能等待追问结束。' if record['risk_level'] == '紧急优先级预警'
                                             else assessment['actions'][0]),
                      'extinguisher_eligible': assessment['extinguisher_eligible']} if assessment else None),
            'references': [{'title': ref['title'], 'excerpt': ref['excerpt'][:140],
                            'review_status': ref.get('review_status')} for ref in record['knowledge_refs'][:2]],
            'confirmed_conditions': {k: v for k, v in record['safety_context'].items() if v is True or k == 'fire_type'},
            'rounds_left': rounds_left,
            'scene_reused': context.get('scene_reused', False), 'scene_memory': memory}


SYSTEM = '''你是离线应急Agent。只处理当前question，history仅帮助理解追问。自主选工具：detect_people查人数位置，detect_fire_smoke查烟火，search_knowledge(query)检索本地资料，assess_risk评估预警。
action=tools时填写tools列表；证据足够时action=finish、answer简短回答，缺信息可另填question追问一句。普通问答无需检测；已完成工具不重复调用。
只能引用工具证据，不能直接看图或编造伤情、出口、起火物。scene_memory通过对应工具复用；历史不能替代新图证据。检测失败是未知，未检出不保证安全。预警等级以risk为准，不能当成火灾概率。
应对问题先检索资料；紧急预警先撤离并在安全位置拨打119，不询问是否需要撤离。灭火条件未确认不能建议扑救。先结论后建议，默认2到4句，遵守用户指定格式，不重复用户问题。'''



def check_answer(text, record):
    """Reject contradicting counts, invented visual claims and unsafe directions."""
    if not text.strip():
        return '回答为空'
    if '<|' in text or re.search(r'已(?:经)?(?:拨打|联系|通知)(?:了)?(?:119|消防|救援)', text):
        return '回答声称执行了未提供的动作'
    if record.get('response_assessment') and any(number not in {'119', '120', '110', '122'}
                                              for number in re.findall(r'拨打\s*(\d+)', text)):
        return '报警电话号码不正确'
    if record['image'] or record.get('response_assessment'):
        if re.search(r'(?:现场|这里|图片|画面).{0,8}(?:绝对安全|完全安全|没有危险|无危险)|可以放心|可以解除警报', text):
            return '不能据图片认定现场安全'
        # This application has no general image-understanding tool.
        if re.search(r'(?:穿着|身穿|受了伤|已经受伤|有人受伤|有人倒地|出口在|起火原因是|起火物是)', text):
            return '包含未被工具证实的图像细节'
        count = record.get('person_statistics', {}).get('estimated_count')
        if count is None and re.search(r'没有人|未检出人员|未发现人员|现场无人', text):
            return '人员检测未完成，不能声称无人'
        for label, name in [('fire', '明火|火焰|火灾|火'), ('smoke', '烟雾|浓烟|烟')]:
            negative = re.search(r'(?:没有|未检出|未发现|不存在)(?:任何|明显)?(?:' + name + ')', text)
            state = record['vision_tools']['fire_smoke']
            positive = any(d['label'] == label for d in record['detections'])
            if negative and (state['status'] != 'ok' or positive):
                return '烟火判断与工具证据不一致'
        numerals = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        for match in re.finditer(r'(\d+|[零一二两三四五六七八九十])\s*(?:个|名|位)?(?:人员|人)(?!工|员(?:工|培训))', text):
            token = match.group(1)
            value = int(token) if token.isdigit() else numerals[token]
            if count is None or value != count:
                return '人数与检测证据不一致'
        score = (record.get('response_assessment') or {}).get('score')
        if any(int(m.group(1)) != score for m in re.finditer(r'(\d+)\s*(?:/\s*100|分)', text)):
            return '评分与预警工具不一致'
        if '概率' in text and re.search(r'\d+\s*[%％]', text):
            return '不能把置信度当作火灾概率'
        assessment = record.get('response_assessment')
        if assessment:
            for match in re.finditer(r'(?:可以|建议|尝试|直接|立即|使用|用).{0,10}(?:扑灭|扑救|灭火器|泼水|浇水)', text):
                prefix = re.split(r'[，,。；;\n]', text[max(0, match.start()-16):match.start()])[-1]
                nearby = prefix + match.group(0)
                if not re.search(r'不要|不可|禁止|勿|不能|不建议|不应|不宜|不具备', nearby):
                    return '具体扑救操作由现场条件和审核规则提供'
            for _, _, label, _ in triage.LEVELS:
                if label in text and label != record['risk_level']:
                    return '预警等级与工具不一致'
    return None


def run_agent(args):
    started = time.perf_counter()
    record = base_record(args)
    context = copy.deepcopy(getattr(args, 'agent_context', None) or {})
    context['detection_allowed'] = bool(args.npu_detect)
    history = context.get('history', [])[-4:]
    record['dialogue_history'] = history
    record['conversation_id'] = context.get('conversation_id')
    record['scene_reused'] = bool(context.get('scene_reused'))
    legacy_plan = agent.request_plan(args.task, bool(args.image), args.observation or '')
    chosen = set()
    calls = set()
    errors = []
    knowledge_queries = set()
    progress_path = getattr(args, 'progress_file', None)

    def trace(title, status='ok', detail='', source='runtime'):
        item = {'title': title, 'status': status, 'detail': detail[:220], 'source': source,
                'at': agent.utc_now()}
        record['agent_trace'].append(item)
        record['events'].append({'step': title, 'status': status, 'at': item['at']})
        print(f'Agent / {title} / {status}: {detail}', flush=True)
        if progress_path:
            path = Path(progress_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            decisions = [{'action': d['action'],
                          'tools': [{'name': t['name']} for t in d.get('tools', [])]}
                         for d in record['agent_execution']['decisions']]
            tmp.write_text(json.dumps({'steps': record['agent_trace'], 'current': title,
                                      'decisions': decisions,
                                      'execution': {'status': record['agent_execution']['status']},
                                      'alert': (agent.fire_smoke_answer(record, agent.fire_smoke_summary(record))
                                                if record.get('risk_level') == '紧急优先级预警' else '')}, ensure_ascii=False))
            tmp.replace(path)

    def sync():
        refresh_evidence(record)
        show_people = ('detect_people' in chosen and (legacy_plan['show_people'] or
                       bool(re.search(r'人|现场|情况|隐患|风险|危险', args.task))))
        show_risk = 'assess_risk' in chosen or 'detect_fire_smoke' in chosen
        plan = {'kind': 'person_count' if show_people else 'risk_assessment' if show_risk else legacy_plan['kind'],
                'use_detection': show_people or 'detect_fire_smoke' in chosen,
                'show_people': show_people, 'show_risk': show_risk,
                'unsupported_visual': legacy_plan['unsupported_visual'],
                'asks_advice': legacy_plan['asks_advice'], 'planner': VERSION}
        record.update(request_plan=plan, answer_version=3, answer_presentation_version=1)
        record.pop('task_answer', None)
        enriched = agent.with_task_result(record)
        record.update(enriched)
        record['evidence_answer'] = record['task_answer'] if show_people or show_risk else None

    def assess():
        chosen.add('assess_risk')
        calls.add(('assess_risk', ''))
        record['response_assessment'] = triage.assess(record, record['safety_context'])
        sync()

    def execute(item, source='qwen'):
        name = item['name']
        query = item.get('query', '').strip() or args.task
        key = (name, query if name == 'search_knowledge' else '')
        if key in calls or (name == 'search_knowledge' and len(knowledge_queries) >= 2):
            trace(TITLES[name], 'skipped', '本轮已取得结果，不重复执行', source)
            return False
        calls.add(key)
        chosen.add(name)
        trace(TITLES[name], 'running', query[:90] if name == 'search_knowledge' else '正在读取本地工具结果', source)
        try:
            if name in {'detect_people', 'detect_fire_smoke'}:
                if not record['image'] or not args.npu_detect:
                    raise ValueError('没有可用图片或用户已关闭图片检测')
                if name == 'detect_fire_smoke' and not agent.CONFIG.get('fire_smoke', {}).get('enabled'):
                    raise ValueError('烟火检测未启用')
                tool_key = 'person' if name == 'detect_people' else 'fire_smoke'
                stamp = fingerprint(name)
                cached = context.get('cached_tools', {}).get(tool_key, {})
                same_scene = context.get('image_sha256') == record['image']['sha256']
                reuse = (same_scene and cached.get('fingerprint') == stamp and cached.get('state', {}).get('status') == 'ok'
                         and not re.search(r'重新检测|重新识别|再检测|重测', args.task))
                if reuse:
                    state = copy.deepcopy(cached['state'])
                    state['reused_from_case_id'] = cached['case_id']
                    record['vision_tools'][tool_key] = state
                else:
                    detector = agent.run_npu_detector if name == 'detect_people' else agent.run_fire_smoke_detector
                    detections, payload = detector(Path(record['image']['path']))
                    payload = dict(payload, detections=detections)
                    sizes = [s['result']['image_size'] for s in record['vision_tools'].values() if s['status'] == 'ok']
                    if sizes and payload['image_size'] != sizes[0]:
                        raise ValueError('工具返回的图片尺寸不一致')
                    record['vision_tools'][tool_key].update(status='ok', result=payload)
                record['agent_tool_fingerprints'][tool_key] = stamp
                refresh_evidence(record)
                if name == 'detect_people':
                    weight = agent.person_model_path()
                    record.update(vision_model_path=str(weight), vision_model_sha256=agent.model_digest(weight))
                else:
                    record['fire_smoke_model_path'] = agent.CONFIG['fire_smoke']['model_dir']
                sync()
                counts = {}
                for d in record['vision_tools'][tool_key]['result']['detections']:
                    counts[d['label']] = counts.get(d['label'], 0) + 1
                trace(TITLES[name], 'reused' if reuse else 'ok',
                      ('沿用同一张图片的已验证结果：' if reuse else '检测完成：') + json.dumps(counts, ensure_ascii=False), source)
                # Always calculate a safety floor as soon as smoke/fire evidence is available.
                if name == 'detect_fire_smoke':
                    assess()
                    trace('即时预警', 'ok', record['risk_level'], 'safety_rule')
            elif name == 'search_knowledge':
                knowledge_queries.add(query)
                db = agent.connect()
                try:
                    refs = agent.retrieve(db, query)
                finally:
                    db.close()
                existing = {(ref['doc_id'], ref['line']) for ref in record['knowledge_refs']}
                record['knowledge_refs'].extend(ref for ref in refs if (ref['doc_id'], ref['line']) not in existing)
                record['knowledge_refs'] = record['knowledge_refs'][-6:]
                trace(TITLES[name], 'ok', f'找到 {len(refs)} 条相关资料' if refs else '本地资料未匹配，不编造引用', source)
            else:
                assess()
                trace(TITLES[name], 'ok', record['risk_level'], source)
            return True
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            if name in {'detect_people', 'detect_fire_smoke'}:
                tool_key = 'person' if name == 'detect_people' else 'fire_smoke'
                record['vision_tools'][tool_key].update(status='failed', error=str(exc)[-600:])
                refresh_evidence(record)
                if name == 'detect_fire_smoke':
                    assess()
                else:
                    sync()
            trace(TITLES[name], 'failed', str(exc)[-200:], source)
            return True

    def required_tools():
        # Enforce evidence for explicit questions if the planner tries to answer without it.
        required = []
        if args.image and args.npu_detect:
            if legacy_plan['show_people'] and 'detect_people' not in chosen:
                required.append({'name': 'detect_people'})
            if legacy_plan['show_risk'] and 'detect_fire_smoke' not in chosen:
                required.append({'name': 'detect_fire_smoke'})
        if legacy_plan['show_risk'] and 'assess_risk' not in chosen:
            required.append({'name': 'assess_risk'})
        if needs_guidance() and record['request_plan']['show_risk'] and 'search_knowledge' not in chosen:
            required.append({'name': 'search_knowledge', 'query': args.task + ' 撤离 报警 现场确认'})
        return required

    def needs_guidance():
        return legacy_plan['asks_advice'] or bool(re.search(r'能(?:不能)?(?:自己|自行|用)|可(?:以|否)|灭火器|怎么办|疏散|撤离', args.task))

    sync()
    trace('理解当前问题', 'running', '本地Qwen选择工具与下一步动作', 'qwen')
    # A follow-up about handling the same incident retains its safety floor.
    # Ordinary questions in that conversation still remain ordinary questions.
    handling_followup = (context.get('scene_reused') and context.get('cached_tools', {}).get('fire_smoke')
                         and re.search(r'灭火|扑救|油锅|燃气|电池|带电|撤离|疏散|报警|被困|怎么办|怎么处理', args.task)
                         and not re.search(r'解释|原理|科普|翻译|写.*|应急包|是什么', args.task))
    if handling_followup and args.npu_detect:
        execute({'name': 'detect_fire_smoke'}, 'memory_guard')
    decision = None
    max_rounds = max(1, min(3, agent.CONFIG.get('agent', {}).get('max_rounds', 3)))
    for round_index in range(max_rounds):
        try:
            trace(f'第 {round_index + 1} 轮规划', 'running', '根据当前问题、已完成工具与会话记录选择下一步', 'qwen')
            schema = SCHEMA
            if (round_index == 0 and args.image and args.npu_detect and legacy_plan['use_detection']
                    and any(item['name'] in {'detect_people', 'detect_fire_smoke'} for item in required_tools())):
                schema = action_schema('tools')
            elif needs_guidance() and record['request_plan']['show_risk'] and 'search_knowledge' not in chosen:
                schema = action_schema('tools')
                schema['properties']['tools']['items']['properties']['name']['enum'] = ['search_knowledge']
            elif round_index > 0 and not required_tools() and chosen and record['knowledge_refs']:
                # The needed evidence and references are present: request synthesis.
                schema = FINAL_SCHEMA
            decision = complete(SYSTEM, planner_state(record, context, max_rounds-round_index), schema=schema,
                                prompt_cache=getattr(args, 'agent_prompt_cache', None))
            record['agent_execution']['decisions'].append(copy.deepcopy(decision))
            selected = ('选择工具：' + '、'.join(TITLES[item['name']] for item in decision['tools'])
                        if decision['action'] == 'tools' else
                        '请求补充信息' if decision['action'] == 'ask' else '提交回答，进入证据校验')
            trace(f'第 {round_index + 1} 轮工具决策', 'ok', selected, 'qwen')
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc)[-600:])
            trace('本地规划', 'failed', str(exc)[-150:], 'qwen')
            decision = None
            break
        if decision['action'] == 'tools':
            changed = False
            for item in decision['tools']:
                changed = execute(item) or changed
            if not changed:
                trace('停止重复调用', 'ok', '进入最终回答', 'runtime')
                decision = None
                break
        else:
            missing = required_tools()
            if missing:
                trace('补齐问题所需证据', 'ok', '回答前验证必需工具已执行', 'evidence_guard')
                for item in missing:
                    execute(item, 'evidence_guard')
                decision = None
                continue
            break
    else:
        decision = None

    for item in required_tools():
        execute(item, 'fallback' if errors else 'evidence_guard')
    sync()
    if decision is None or decision['action'] == 'tools':
        try:
            trace('整理回答', 'running', '结合工具证据与本地资料生成结论', 'qwen')
            decision = complete(SYSTEM + '\n现在结束工具调用，只能finish或ask。依据已有证据简短回答；没有取得证据的内容明确未知。',
                                planner_state(record, context, 0), schema=FINAL_SCHEMA,
                                prompt_cache=getattr(args, 'agent_prompt_cache', None))
            record['agent_execution']['decisions'].append(copy.deepcopy(decision))
            trace('最终回答决策', 'ok', '提交回答，进入证据校验', 'qwen')
            if decision['action'] == 'tools':
                raise ValueError('达到工具轮次上限，停止继续调用')
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc)[-600:])
            decision = None

    candidate = (decision or {}).get('answer', '').strip()
    question = (decision or {}).get('question', '').strip()
    if (record['risk_level'] == '紧急优先级预警' and
            re.search(r'(?:是否|需要|要不要|需不需要).{0,8}(?:紧急撤离|报警|危险)', question)):
        trace('追问校验', 'rejected', '紧急避险已明确，不再追问是否需要避险', 'evidence_guard')
        question = ''
    if question and (re.search(r'几个人|有多少人|有火吗|有没有(?:烟|火)', question) or
                     question.strip('？?。 ') == args.task.strip('？?。 ')):
        trace('追问校验', 'rejected', '不重复询问用户问题和已取得的检测结果', 'evidence_guard')
        question = ''
    record['agent_draft'] = candidate
    problem = check_answer(candidate + ('\n' + question if question else ''), record) if candidate or question else None
    if problem:
        trace('证据校验', 'rejected', problem + '，使用工具结论', 'evidence_guard')
        candidate = ''
        question = ''
    # Qwen writes the answer. Append only missing authoritative facts/guidance.
    header = []
    if record['request_plan']['show_people']:
        stats = record['person_statistics']
        count = stats['estimated_count']
        has_count = bool(re.search(r'(\d+|[零一二两三四五六七八九十])\s*(?:个|名|位)?人', candidate))
        if not candidate or not has_count:
            header.append(f'图片中估计有 {count} 人。' if count is not None and count > 0 else
                          '图片中未检出人员，不能确认现场无人。' if count == 0 else '人员检测未完成，人数未知。')
    if record['request_plan']['show_risk']:
        evidence, grade, *actions = agent.fire_smoke_answer(record, agent.fire_smoke_summary(record)).split('\n')
        if not candidate:
            header.append('\n'.join([evidence, grade, *actions]))
        else:
            state = record['vision_tools']['fire_smoke']
            fire_found = any(d['label'] == 'fire' for d in record['detections'])
            smoke_found = any(d['label'] == 'smoke' for d in record['detections'])
            if (state['status'] != 'ok' or fire_found and not re.search(r'明火|火焰|着火', candidate)
                    or smoke_found and not re.search(r'烟雾|浓烟', candidate)):
                header.append(evidence)
            assessment = record.get('response_assessment')
            if assessment and not (record['risk_level'] in candidate and f"{assessment['score']}/100" in candidate):
                header.append(grade)
            if assessment and assessment['level'] == 0 and not re.search(r'漏检|不代表安全|不能.*安全|无法.*排除', candidate):
                header.append('未检出不代表安全，请在安全位置复核。')
            if actions and (not re.search(r'撤离|保持距离|安全位置.*核查', candidate) or
                            record['risk_level'] == '紧急优先级预警' and '119' not in candidate):
                header.extend(actions)
    if not candidate and not header:
        candidate = ('当前工具只能识别人员、烟雾和明火，无法从图片判断你询问的其他细节。请补充文字描述。'
                     if legacy_plan['kind'] == 'visual_question' else '本地模型未能完成本次回答，请重试或补充具体问题。')
    answer = '\n'.join(([candidate] if candidate else []) + header)
    if question and question not in answer:
        answer += ('\n' if answer else '') + question
    if record.get('knowledge_refs') and header and not candidate:
        answer += '\n相关处置资料已列在“检测详情与参考资料”中。'
    record.update(task_answer=answer.strip(), llm_explanation=None,
                  llm_error=('；'.join(errors) or None) if decision is None else None, follow_up_question=question or None,
                  answer_source='local_agent' if decision and not problem else 'agent_evidence_fallback',
                  answer_presentation_version=1, answer_version=3,
                  elapsed_seconds=round(time.perf_counter()-started, 3))
    record['agent_execution'].update(status='completed' if decision and not problem else 'fallback',
                                     rounds=len(record['agent_execution']['decisions']), errors=errors)
    trace('等待补充信息' if question else '回答完成', 'ok', '已保存本轮工具依据', 'runtime')
    record['elapsed_seconds'] = round(time.perf_counter()-started, 3)
    agent.save_record(record)
    return record

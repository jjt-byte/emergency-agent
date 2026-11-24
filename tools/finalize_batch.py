#!/usr/bin/env python3
"""Export completed results and attach explicit knowledge/report provenance."""
from collections import Counter
import csv
import html
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
import triage


def image_presence_metrics(entries):
    """Evaluate whether each class was found anywhere in an annotated image."""
    metrics = {scope: {label: dict(tp=0, fp=0, fn=0, tn=0, ignored=0) for label in ('fire', 'smoke')}
               for scope in ('all_labeled', 'val')}
    for entry in entries:
        if not entry.get('evaluation'):
            continue
        for scope in ['all_labeled'] + (['val'] if entry['split'] == 'val' else []):
            for label, evidence in entry['evaluation'].items():
                present = evidence['tp'] + evidence['fn'] > 0
                found = any(d['label'] == label for d in entry.get('detections', []))
                if evidence.get('ignored_difficult') and not present:
                    metrics[scope][label]['ignored'] += 1
                    continue
                key = 'tp' if present and found else 'fn' if present else 'fp' if found else 'tn'
                metrics[scope][label][key] += 1
    for classes in metrics.values():
        for m in classes.values():
            tp, fp, fn = m['tp'], m['fp'], m['fn']
            m.update(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                     f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return metrics


def finalize(batch_id):
    folder = agent.DATA/'batches'/batch_id
    status = json.loads((folder/'status.json').read_text())
    if status.get('status') != 'done' or status.get('completed') != status.get('total'):
        raise ValueError('批次未完成，不能导出完成报告')
    entries = [json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines()]
    if len(entries) != status['total'] or len({e['index'] for e in entries}) != status['total']:
        raise ValueError('结果条目缺失或重复')
    db = agent.connect()
    attached = updated = 0
    for entry in entries:
        if not entry.get('case_id'):
            continue
        record = json.loads((agent.REPORTS/(entry['case_id']+'.json')).read_text())
        record['dataset_sample'] = {'batch_id': batch_id, 'name': entry['name'], 'archive': entry['archive'],
                                    'split': entry['split'], 'source': status['dataset_source']}
        refs = agent.retrieve(db, record['task'] + ' 烟雾 明火 报警 撤离 灭火器 评分')
        if record.get('knowledge_refs') != refs:
            if not record.get('knowledge_refs'):
                attached += 1
            else:
                record.setdefault('initial_knowledge_refs', record['knowledge_refs'])
            record['knowledge_refs'] = refs
            record.setdefault('events', []).append({'step': 'attach_knowledge_after_batch', 'status': 'ok', 'at': agent.utc_now()})
            updated += 1
        for ref in record['knowledge_refs']:
            if str(ref['source']).startswith(str(agent.ROOT/'knowledge')+'/'):
                ref['review_status'] = 'pending_local_review'
        record['batch_evaluation'] = {key: entry.get(key) for key in ('evaluation', 'ground_truth_count', 'annotation_error')}
        agent.save_record(record)
    db.close()
    status.update(report_finalized_at=agent.utc_now(), knowledge_attached_after_inference=attached, knowledge_updated_after_inference=updated,
        annotation_quality_note='抽查发现部分可见烟雾没有标注，标注粒度也不同。指标只表示与所提供标注的匹配，不等同于全部真实烟火目标的误检与漏检。',
        triage_notice='实验性图片预警分，不是火灾概率或真实火势等级；不能仅凭分数决定自行灭火。',
        knowledge_status='6类项目整理提示已离线导入；机构原文尚未联网核验，待现场审核。')
    status['image_presence_metrics'] = image_presence_metrics(entries)
    status['score_calibration_status'] = '数据只有烟火目标框，没有实际火势等级或扑救许可标签；本轮不能验证四档预警分的风险准确率。'
    for name in ['summary.json']:
        temporary = folder/(name+'.tmp')
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(folder/name)
    counts = Counter(str(e.get('level')) for e in entries)
    lines = [f'# 数据集烟火检测报告 {batch_id}', '', f"来源：{status['dataset_source']}", '',
        f"共{status['total']}张；已处理{status['completed']}张，工具错误{status['errors']}张，有效标注评估{status.get('evaluated',0)}张，标注无效{status.get('annotation_errors',0)}张。", '',
        status['triage_notice'], '', '| 预警分 | 等级 | 图片数 | 应对优先级 |', '|---|---|---:|---|']
    for i, (low, high, label, priority) in enumerate(triage.LEVELS):
        lines.append(f'| {low}–{high} | {label} | {counts[str(i)]} | {priority} |')
    lines += ['', f"未知未评分：{counts['None']}张；0分表示模型未检出，不代表安全。", '',
              '| 范围 | 类别 | Precision | Recall | F1 | TP / FP / FN |', '|---|---|---:|---:|---:|---|']
    fmt = lambda value: f'{value:.2%}' if value is not None else '无分母'
    for scope, classes in status['metrics'].items():
        for label, m in classes.items():
            lines.append(f"| {scope} | {label} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {m['tp']} / {m['fp']} / {m['fn']} |")
    lines += ['', '图片级有无检测（只判断该类是否在图片任意位置检出，不要求框位置匹配）：', '',
              '| 范围 | 类别 | Precision | Recall | F1 | TP / FP / FN / TN |', '|---|---|---:|---:|---:|---|']
    for scope, classes in status['image_presence_metrics'].items():
        for label, m in classes.items():
            lines.append(f"| {scope} | {label} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']} |")
    lines += ['', status['scope'], '', status['annotation_quality_note'], '',
        '仅计算置信度阈值0.6、IoU≥0.5的框匹配指标，不是mAP。包括train列表，val另行报告，但与模型预训练数据重叠未知，不能作为独立泛化测试。无标注634张只做推理。', '', status['score_calibration_status'], '',
        '自行灭火须额外满足全部现场条件，且不属于紧急预警；默认未确认时不批准。现场确认发生火灾应立即报警，不等待模型。油锅禁止泼水，未确认断电设备不用水；燃气、电池和化学品危险交专业救援。', '', status['knowledge_status']]
    markdown = '\n'.join(lines)+'\n'
    (folder/'summary.md').write_text(markdown)
    fields = ['index','name','archive','split','score','level','label','case_id','fire_confidence','smoke_confidence',
              'fire_tp','fire_fp','fire_fn','smoke_tp','smoke_fp','smoke_fn','error','annotation_error']
    with (folder/'results.csv.tmp').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
        for entry in entries:
            row = {key: entry.get(key, '') for key in fields}
            for label in ('fire', 'smoke'):
                row[label+'_confidence'] = max((d['confidence'] for d in entry.get('detections', []) if d['label'] == label), default='')
                for key in ('tp', 'fp', 'fn'):
                    row[label+'_'+key] = entry.get('evaluation', {}).get(label, {}).get(key, '')
            writer.writerow(row)
    (folder/'results.csv.tmp').replace(folder/'results.csv')
    rows = []
    for entry in entries:
        name = html.escape(entry['name']);case = entry.get('case_id')
        link = f'<a href="../../reports/{case}.html">{name}</a>' if case else name
        rows.append(f"<tr><td>{link}</td><td>{entry.get('score')}</td><td>{html.escape(entry['label'])}</td><td>{html.escape(entry['split'])}</td></tr>")
    (folder/'summary.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>烟火批量报告</title><style>body{font:15px sans-serif;max-width:1200px;margin:24px auto;padding:15px;line-height:1.8}pre{white-space:pre-wrap}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{width:100%}a{color:#19614e}</style><pre>'+html.escape(markdown)+'</pre><p><a href="results.csv">完整CSV</a></p><table><tr><th>图片 / 检测框与应对建议</th><th>预警分</th><th>等级</th><th>拆分</th></tr>'+''.join(rows)+'</table></html>')
    temporary = folder/'status.final.tmp'
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(folder/'status.json')
    print(f'已导出{len(entries)}张图片报告和CSV，补挂知识依据{attached}张，更新引用{updated}张。')
    return status


if __name__ == '__main__':
    import argparse,re
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('batch_id');args=parser.parse_args()
    if not re.fullmatch(r'\d{8}-\d{6}', args.batch_id):parser.error('批次编号无效')
    finalize(args.batch_id)

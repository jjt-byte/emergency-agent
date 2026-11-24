#!/usr/bin/env python3
"""Idempotently import the visibly labeled project-curated knowledge notes."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent


def main():
    db = agent.connect()
    sources = []
    for path in sorted((agent.ROOT/'knowledge').glob('*.md')):
        body = path.read_text()
        title = body.splitlines()[0].lstrip('# ')
        source = str(path.resolve())
        existing = db.execute('SELECT id FROM docs WHERE source=?', (source,)).fetchone()
        if existing:
            db.execute('UPDATE docs SET title=?,body=?,added_at=? WHERE id=?', (title, body, agent.utc_now(), existing[0]))
        else:
            db.execute('INSERT INTO docs(title,source,body,added_at) VALUES(?,?,?,?)', (title, source, body, agent.utc_now()))
        sources.append({'title': title, 'path': source, 'sha256': agent.sha256(path),
                        'provenance': 'project_curated', 'review_status': 'pending_local_review',
                        'official_reference_downloaded': False})
    db.commit()
    db.close()
    (agent.DATA/'knowledge_manifest.json').write_text(json.dumps({'version': 1, 'imported_at': agent.utc_now(),
        'note': '网络DNS不可用；这些是项目整理提示，不冒充已下载核实的官方原文。', 'documents': sources}, ensure_ascii=False, indent=2))
    print(f'已导入或更新 {len(sources)} 条本地资料，均标明待审核。')


if __name__ == '__main__':
    main()

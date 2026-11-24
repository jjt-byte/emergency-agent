#!/usr/bin/env python3
"""Discover public Hugging Face fire/smoke weights and download without loading them.

Downloaded weights remain unvalidated candidates, never active inference models.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

BASE = 'https://huggingface.co'
DEFAULT_DIR = Path(__file__).resolve().parents[1] / 'models' / 'fire-smoke'
OPEN_LICENSES = {'apache-2.0', 'mit', 'bsd-2-clause', 'bsd-3-clause', 'cc0-1.0', 'cc-by-4.0',
                 'agpl-3.0', 'gpl-3.0', 'gpl-3.0-only', 'gpl-3.0-or-later'}
USER_AGENT = 'offline-emergency-agent-model-downloader/1.0'


def fetch(url, binary=False):
    with urlopen(Request(url, headers={'User-Agent': USER_AGENT}), timeout=30) as response:
        body = response.read(5 * 1024 * 1024)
    return body if binary else json.loads(body)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def model_info(repo):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('仓库名称必须是 author/model 格式')
    return fetch(BASE + '/api/models/' + quote(repo, safe='/') + '?blobs=true')


def list_candidates(destination):
    repositories = {}
    errors = []
    for search in ['fire-smoke', 'fire smoke', 'fire_smoke']:
        try:
            models = fetch(BASE + '/api/models?' + urlencode({'search': search, 'limit': 25, 'sort': 'downloads', 'direction': -1}))
            for model in models:
                name = model.get('id', model.get('modelId', ''))
                if name:
                    repositories[name] = model
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    candidates = []
    for repo in list(repositories)[:30]:
        try:
            info = model_info(repo)
            if info.get('private') or info.get('gated'):
                continue
            if info.get('pipeline_tag') and info['pipeline_tag'] != 'object-detection':
                continue
            card = info.get('cardData') or {}
            license_name = card.get('license', '')
            if isinstance(license_name, list):
                license_name = ', '.join(license_name)
            text = (repo + ' ' + json.dumps(card, ensure_ascii=False)).lower()
            if 'fire' not in text or 'smoke' not in text:
                continue
            if not any(term in text for term in ('yolo', 'detection', 'object-detection')) and info.get('pipeline_tag') != 'object-detection':
                continue
            for artifact in info.get('siblings', []):
                name = artifact['rfilename']
                if PurePosixPath(name).suffix.lower() not in {'.onnx', '.pt'}:
                    continue
                size = artifact.get('size') or (artifact.get('lfs') or {}).get('size')
                if size and size > 1024 * 1024 * 1024:
                    continue
                revision = info.get('sha')
                if not revision:
                    continue
                candidates.append({'repo': repo, 'file': name, 'revision': revision,
                                   'license': license_name, 'downloads': info.get('downloads', 0),
                                   'size_bytes': size, 'sha256_expected': (artifact.get('lfs') or {}).get('sha256'),
                                   'model_page': BASE + '/' + repo,
                                   'download_url': BASE + '/' + repo + '/resolve/' + revision + '/' + quote(name, safe='/')})
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{repo}: {exc}')
    candidates.sort(key=lambda item: (item['license'] not in OPEN_LICENSES,
                                     not item['file'].lower().endswith('.onnx'),
                                     'best' not in item['file'].lower(), -item['downloads']))
    write_json(destination / 'candidates.json', {'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                                               'candidates': candidates, 'errors': errors})
    return candidates, errors


def download(candidate, destination):
    name = PurePosixPath(candidate['file']).name
    if name in {'', '.', '..'}:
        raise ValueError('权重文件名无效')
    directory = destination / candidate['repo'].replace('/', '--')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    part = target.with_name(target.name + '.part')
    digest = hashlib.sha256()
    total = 0
    try:
        with urlopen(Request(candidate['download_url'], headers={'User-Agent': USER_AGENT}), timeout=60) as response:
            with part.open('wb') as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 1024 * 1024 * 1024:
                        raise ValueError('模型超过 1 GB，已停止下载')
                    output.write(chunk)
                    digest.update(chunk)
        expected = candidate.get('sha256_expected')
        if expected and expected != digest.hexdigest():
            raise ValueError('SHA256 与仓库声明不一致')
        if candidate.get('size_bytes') and total != candidate['size_bytes']:
            raise ValueError('下载大小与仓库声明不一致')
        with part.open('rb') as downloaded:
            head = downloaded.read(256).lstrip()
        if not total or head.startswith((b'<html', b'<!DOCTYPE', b'version https://git-lfs')):
            raise ValueError('下载内容不是模型权重')
        part.replace(target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    provenance = dict(candidate, local_path=str(target.resolve()), actual_sha256=digest.hexdigest(),
                      actual_size_bytes=total, status='downloaded_unvalidated',
                      downloaded_at=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(directory / 'source.json', provenance)
    for remote, local in [('README.md', 'MODEL_CARD.md'), ('LICENSE', 'UPSTREAM-LICENSE')]:
        try:
            url = BASE + '/' + candidate['repo'] + '/resolve/' + candidate['revision'] + '/' + remote
            (directory / local).write_bytes(fetch(url, binary=True))
        except OSError:
            pass
    links = (f"# 烟火候选模型来源\n\n模型页面：{candidate['model_page']}\n\n"
             f"权重下载链接：{candidate['download_url']}\n\n本地文件：{target.resolve()}\n\n"
             f"SHA256：{digest.hexdigest()}\n\n许可证声明：{candidate.get('license') or '未声明'}\n\n"
             "状态：已下载，类别、输出格式、实际精度和 CIX NPU 兼容性尚未验证。\n"
             "ONNX 需验证输入输出和运行依赖；PT 需安全加载、核对类别、导出 ONNX。\n"
             "转换为 .cix 前需经官方工具链编译和板端实测，当前网页不会自动启用这些候选权重。\n")
    (directory / 'DOWNLOAD_LINKS.md').write_text(links, encoding='utf-8')
    print(links)
    return provenance


def main():
    parser = argparse.ArgumentParser(description='检索并下载烟雾/明火候选模型，不执行下载的代码')
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--auto', action='store_true', help='下载一个公开许可证的候选，优先 ONNX')
    parser.add_argument('--repo', help='指定模型仓库 author/model')
    parser.add_argument('--file', help='指定仓库内权重文件名')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.repo:
        info = model_info(args.repo)
        files = [f for f in info.get('siblings', []) if PurePosixPath(f['rfilename']).suffix.lower() in {'.onnx', '.pt'}]
        if not args.file:
            print('仓库权重文件：\n' + '\n'.join(f['rfilename'] for f in files))
            return
        artifact = next((f for f in files if f['rfilename'] == args.file), None)
        if not artifact:
            parser.error('指定权重未出现在仓库 API 文件列表中')
        revision = info['sha']
        candidate = {'repo': args.repo, 'file': args.file, 'revision': revision,
                     'license': (info.get('cardData') or {}).get('license', ''),
                     'size_bytes': artifact.get('size') or (artifact.get('lfs') or {}).get('size'),
                     'sha256_expected': (artifact.get('lfs') or {}).get('sha256'),
                     'model_page': BASE + '/' + args.repo,
                     'download_url': BASE + '/' + args.repo + '/resolve/' + revision + '/' + quote(args.file, safe='/')}
        download(candidate, args.directory)
        return
    candidates, errors = list_candidates(args.directory)
    for index, candidate in enumerate(candidates, 1):
        print(f"{index}. {candidate['repo']} / {candidate['file']} [{candidate['license'] or '许可证未声明'}]")
        print('   ' + candidate['download_url'])
    if not candidates:
        raise RuntimeError('未取得可核实的候选模型。' + ('；网络错误：' + errors[0] if errors else '请检查检索结果。'))
    if args.auto:
        eligible = [c for c in candidates if c['license'] in OPEN_LICENSES]
        if not eligible:
            raise RuntimeError('候选未声明可识别的公开许可证，已保存候选链接，请先核对来源。')
        download(eligible[0], args.directory)
    else:
        print(f'候选和链接已保存：{args.directory / "candidates.json"}；使用 --auto 下载一个候选。')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print('检索/下载未完成：' + str(exc), file=sys.stderr)
        sys.exit(1)

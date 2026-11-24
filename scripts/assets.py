#!/usr/bin/env python3
"""Import/verify separately supplied assets; never deserialize model weights."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def entries(kind):
    return json.loads((ROOT / f'benchmarks/{kind}_manifest.json').read_text())['files']


def safe_target(name, kind):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'非法资源路径: {name}')
    allowed = ('models',) if kind == 'models' else ('samples',)
    target = ROOT.joinpath(*path.parts)
    if path.parts[0] not in allowed or not target.resolve().is_relative_to(ROOT):
        raise ValueError(f'资源路径越界: {name}')
    return target


def import_archive(archive, kind):
    manifest = {x['path']: x for x in entries(kind)}
    with tarfile.open(archive, 'r:*') as source:
        members = source.getmembers()
        found = set()
        for member in members:
            if member.isdir():
                continue
            if not member.isfile() or member.name not in manifest or member.name in found:
                raise ValueError(f'归档包含未声明文件/链接/重复文件: {member.name}')
            expected = manifest[member.name]
            if member.size != expected['bytes']:
                raise ValueError(f'资源大小不匹配: {member.name}')
            found.add(member.name)
        if found != set(manifest):
            raise ValueError('归档缺少资源: ' + ', '.join(sorted(set(manifest)-found)))
        for member in members:
            if not member.isfile():
                continue
            target = safe_target(member.name, kind)
            if target.is_file() and sha256(target) == manifest[member.name]['sha256']:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.part')
            try:
                with source.extractfile(member) as incoming, temporary.open('wb') as output:
                    shutil.copyfileobj(incoming, output, 4 * 1024 * 1024)
                if sha256(temporary) != manifest[member.name]['sha256']:
                    raise ValueError(f'资源 SHA256 不匹配: {member.name}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def verify(kind):
    for item in entries(kind):
        target = safe_target(item['path'], kind)
        if not target.is_file() or target.stat().st_size != item['bytes'] or sha256(target) != item['sha256']:
            raise ValueError('缺失或校验失败: ' + item['path'])
    print(f'{kind}: SHA256 校验通过，{len(entries(kind))} 个文件')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['models', 'samples'])
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.archive:
        import_archive(args.archive, args.kind)
    verify(args.kind)

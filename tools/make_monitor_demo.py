#!/usr/bin/env python3
"""Create a clearly labelled image-sequence demo; it is not surveillance footage."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monitor import now, probe_video, write_json


def original_demo():
    directory = ROOT / 'data' / 'monitor' / 'videos'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'demo.mp4'
    samples = [ROOT / 'samples/validation-blank.png', ROOT / 'samples/000000559842.jpg',
               ROOT / 'samples/fire-smoke/ESE_fire_10.jpg', ROOT / 'samples/validation-blank.png']
    with tempfile.TemporaryDirectory(prefix='monitor-demo-') as temporary:
        clips = []
        for index, image in enumerate(samples):
            clip = Path(temporary) / f'{index}.mp4'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-loop', '1', '-i', str(image),
                            '-t', '8', '-vf', 'scale=960:640:force_original_aspect_ratio=decrease,pad=960:640:(ow-iw)/2:(oh-ih)/2:color=white',
                            '-r', '5', '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', '-y', str(clip)], check=True)
            clips.append(clip)
        listing = Path(temporary) / 'clips.txt'
        listing.write_text('\n'.join(f"file '{clip}'" for clip in clips))
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                        '-c', 'copy', '-movflags', '+faststart', '-y', str(output)], check=True)
    item = dict(probe_video(output), id='demo', name='模拟演示 · 空场 → 人员 → 烟火 → 空场',
                filename=output.name, kind='image_sequence_demo', size_bytes=output.stat().st_size,
                imported_at=now(), note='由已有静态测试图片拼接；用于验证流程，不代表真实视频识别效果。')
    write_json(directory / 'demo.json', item)
    print(output)


def demo_suite():
    """Render repeatable scene changes, with source timelines for each video."""
    directory = ROOT / 'data' / 'monitor' / 'videos'
    directory.mkdir(parents=True, exist_ok=True)
    assets = ROOT / 'samples' / 'monitor-suite'
    assets.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / 'data/datasets/smoke_fire_datasets_1/fire_smoke/fire_smoke/images'
    # Copy a small, fixed source set so future dataset cleanup cannot break regeneration.
    import shutil
    photos = {
        'empty': ('空场：无人厨房', ROOT/'samples/000000037777.jpg', []),
        'blank': ('空场：纯色负样例', ROOT/'samples/validation-blank.png', []),
        'people3': ('人员：三人照片', ROOT/'samples/000000252219.jpg', ['person']),
        'people6': ('人员：六人照片', ROOT/'samples/monitor-suite/people6.jpg', ['person']),
        'smoke': ('烟雾照片', dataset/'smoke_000569.jpg', ['smoke']),
        'fire': ('明火照片', dataset/'fire_fire_373.jpg', ['fire']),
        'both': ('烟雾与明火照片', ROOT/'samples/fire-smoke/ESE_fire_10.jpg', ['fire', 'smoke']),
    }
    scenes = {}
    for key, (label, original, labels) in photos.items():
        destination = assets / (key + original.suffix.lower())
        if original.is_file():
            if not destination.exists() or hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                shutil.copyfile(original, destination)
        elif not destination.is_file():
            raise FileNotFoundError('缺少本地演示照片：' + str(original))
        scenes[key] = {'label': label, 'image': str(destination.relative_to(ROOT)),
                       'original_image': str(original), 'labels_present': labels,
                       'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}

    plans = [
        ('01', '基础轮巡', [('empty',8),('people3',8),('both',8),('empty',8),
                          ('people6',8),('smoke',8),('blank',8),('fire',8),('empty',8)],
         '每场景 8 秒；默认 2 秒采样时空场提供约四次未检出，可观察事件结束和再次提醒。'),
        ('02', '烟火分阶段', [('empty',10),('smoke',10),('fire',10),('both',10),
                            ('empty',10),('smoke',10),('empty',10)],
         '仅切换空场、烟雾、明火与烟火照片，用于查看不同类别的首次和连续提醒。'),
        ('03', '快速切换', [('empty',4),('people3',4),('fire',4),('empty',4),('smoke',4),
                          ('people6',4),('blank',4),('both',4),('people3',4),('smoke',4),
                          ('empty',4),('people6',4),('fire',4),('both',4),('empty',4)],
         '每场景 4 秒；2 秒采样时短空场可能不足三次未检出，事件保持活动属于既定合并规则。'),
        ('04', '人数变化与烟火', [('empty',8),('people3',8),('people6',8),('empty',8),('fire',8),
                              ('blank',8),('people3',8),('smoke',8),('both',8),('empty',8)],
         '三人与六人照片交替，穿插空场及烟火；人数是照片的人工点数，模型输出以实际结果为准。'),
        ('05', '两分钟反复切换', [('empty',8),('people3',8),('empty',8),('smoke',8),('empty',8),
                                ('fire',8),('empty',8),('people6',8),('blank',8),('both',8),
                                ('empty',8),('people3',8),('empty',8),('fire',8),('empty',8)],
         '多次从目标照片切回空场，再出现人员和烟火，用于观察重复事件、空场结束和循环回放。'),
    ]
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font.is_file():
        result = subprocess.run(['fc-match', '-f', '%{file}', 'sans-serif'],
                                capture_output=True, text=True, check=True)
        font = Path(result.stdout.strip())
    manifest = {'created_at': now(), 'kind': 'image_sequence_demo', 'fps': 10,
                'resolution': [960, 640], 'scenes': scenes,
                'note': '本地静态照片硬切拼接；画面中的场景名是素材内容，不是模型判断；不是真实连续拍摄或准确率测试。',
                'videos': []}
    with tempfile.TemporaryDirectory(prefix='monitor-suite-') as temporary:
        work = Path(temporary)
        for number, title, schedule, purpose in plans:
            source_id = uuid.uuid5(uuid.NAMESPACE_URL, 'emergency-agent:photo-demo-suite:v1:' + number).hex
            clips, timeline = [], []
            offset = 0
            print(f'制作 {number} {title}…', flush=True)
            for index, (key, seconds) in enumerate(schedule):
                scene = scenes[key]
                clip = work / f'{number}-{index}.mp4'
                text = work / f'{number}-{index}.txt'
                text.write_text(f'照片演示 {number} · {title}  |  {index+1}/{len(schedule)}  {scene["label"]}', encoding='utf-8')
                footer = work / f'{number}-{index}-footer.txt'
                footer.write_text(f'静态照片拼接 · 场景切换 {offset:03d}–{offset+seconds:03d} 秒 · 非实时监控', encoding='utf-8')
                filters = (
                    'scale=960:552:force_original_aspect_ratio=decrease:force_divisible_by=2,'
                    'pad=960:640:(ow-iw)/2:44+(552-ih)/2:color=0x142c29,setsar=1,'
                    f'drawtext=fontfile={font}:textfile={text}:fontcolor=white:fontsize=21:x=20:y=11,'
                    f'drawtext=fontfile={font}:textfile={footer}:fontcolor=0xb6cdc4:fontsize=17:x=20:y=609'
                )
                subprocess.run(['ffmpeg','-nostdin','-v','error','-loop','1','-framerate','10',
                                '-i',str(ROOT/scene['image']),'-t',str(seconds),'-vf',filters,
                                '-an','-r','10','-c:v','libx264','-preset','veryfast','-crf','20',
                                '-threads','2','-pix_fmt','yuv420p','-y',str(clip)],check=True,timeout=60)
                clips.append(clip)
                timeline.append({'index': index+1, 'start_seconds': offset, 'end_seconds': offset+seconds,
                                 'scene': key, **scene})
                offset += seconds
            listing = work / f'{number}-clips.txt'
            listing.write_text('\n'.join(f"file '{clip}'" for clip in clips), encoding='utf-8')
            output = directory / (source_id + '.mp4')
            temporary_output = work / (source_id + '.mp4')
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),
                            '-c','copy','-movflags','+faststart','-y',str(temporary_output)],check=True,timeout=60)
            info = probe_video(temporary_output)
            if abs(info['duration_seconds']-offset) > .1:
                raise RuntimeError('视频时长校验失败：' + title)
            shutil.copyfile(temporary_output, output)
            item = dict(info, id=source_id, name=f'照片演示 {number} · {title} · {offset}秒',
                        filename=output.name, kind='image_sequence_demo', size_bytes=output.stat().st_size,
                        imported_at=now(), note=manifest['note'], purpose=purpose,
                        fps=10, timeline=timeline,
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            write_json(directory / (source_id + '.json'), item)
            manifest['videos'].append(item)
            print(f'已导入 {item["name"]}：{output}', flush=True)
    write_json(ROOT/'data/monitor/photo-demo-suite.json', manifest)
    print('照片演示已全部生成；场景时刻表：data/monitor/photo-demo-suite.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', action='store_true', help='制作五段不同顺序和停留时长的照片演示视频')
    args = parser.parse_args()
    if args.suite:
        demo_suite()
    else:
        original_demo()


if __name__ == '__main__':
    main()

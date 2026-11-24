#!/usr/bin/env python3
"""Install application-menu and desktop launchers for the current user."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
APP_ID = 'cn.yingjixianxing.Desktop'


def main():
    launcher = ROOT / 'start-desktop.sh'
    launcher.chmod(0o755)
    applications = Path.home() / '.local/share/applications'
    applications.mkdir(parents=True, exist_ok=True)
    icons = Path.home() / '.local/share/icons/hicolor/scalable/apps'
    icons.mkdir(parents=True, exist_ok=True)
    icon = icons / (APP_ID + '.svg')
    shutil.copyfile(ROOT / 'desktop/app-icon.svg', icon)
    command = str(launcher)
    for character in (chr(92), chr(34), '$', chr(96)):
        command = command.replace(character, chr(92) + character)
    content = chr(10).join([
        '[Desktop Entry]', 'Type=Application', 'Name=应急先行',
        'Comment=离线监测、现场分析与应急证据管理',
        'Exec=' + chr(34) + command.replace('%', '%%') + chr(34),
        'TryExec=' + str(launcher), 'Icon=' + str(icon),
        'Terminal=false', 'StartupNotify=true', 'Categories=Utility;GTK;', '',
    ])
    target = applications / (APP_ID + '.desktop')
    target.write_text(content, encoding='utf-8')
    target.chmod(0o755)
    desktop = Path.home() / 'Desktop'
    if shutil.which('xdg-user-dir'):
        result = subprocess.run(['xdg-user-dir', 'DESKTOP'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            candidate = Path(result.stdout.strip())
            if candidate.is_relative_to(Path.home()) and candidate != Path.home():
                desktop = candidate
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut = desktop / '应急先行.desktop'
    shortcut.write_text(content, encoding='utf-8')
    shortcut.chmod(0o755)
    if shutil.which('desktop-file-validate'):
        subprocess.run(['desktop-file-validate', str(target)], check=True)
    if shutil.which('update-desktop-database'):
        subprocess.run(['update-desktop-database', str(applications)], check=True)
    if shutil.which('gio'):
        subprocess.run(['gio', 'set', str(shortcut), 'metadata::trusted', 'true'], capture_output=True)
    print('应用菜单：' + str(target))
    print('桌面快捷方式：' + str(shortcut))
    print('若提示未受信任，请右键快捷方式选择“允许启动”。')


if __name__ == '__main__':
    main()

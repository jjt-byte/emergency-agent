#!/usr/bin/env bash
# Run from a terminal in the board's graphical desktop session.
set -eu
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_log_dir="$task_root/data/input-method"
mkdir -p "$task_log_dir"
task_log="$task_log_dir/launcher.log"
ulimit -c 0

if [[ -z "${DISPLAY:-}" ]]; then
    echo '请在板卡图形桌面里打开终端运行此脚本；当前没有 X11 / XWayland 显示连接。'
    exit 1
fi
if ! dbus-send --session --print-reply --dest=org.freedesktop.DBus \
    /org/freedesktop/DBus org.freedesktop.DBus.ListNames > /dev/null 2>&1; then
    echo '无法连接桌面会话 D-Bus。请使用板卡桌面的终端运行，不要使用 sudo。'
    exit 1
fi

export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMODIFIERS=@im=fcitx
{
    date -Is
    echo "Session: ${XDG_SESSION_TYPE:-unknown}; DISPLAY: $DISPLAY"
    echo 'GTK_IM_MODULE=fcitx; QT_IM_MODULE=fcitx; XMODIFIERS=@im=fcitx'
} >> "$task_log"

if [[ "${1:-}" == '--diagnose' ]]; then
    echo '输入法状态（0：不可用，1：英文，2：中文）：'
    fcitx5-remote 2>&1 | tee -a "$task_log"
    echo '当前输入法：'
    fcitx5-remote -n 2>&1 | tee -a "$task_log"
    echo 'Fcitx5 D-Bus 名称：'
    dbus-send --session --print-reply --dest=org.freedesktop.DBus \
        /org/freedesktop/DBus org.freedesktop.DBus.NameHasOwner \
        string:org.fcitx.Fcitx5 2>&1 | tee -a "$task_log"
    echo "日志：$task_log"
    exit 0
fi

# Replace the input method daemon in this user's desktop session only.
fcitx5 -rd --ui=classicui >> "$task_log" 2>&1
task_ready=false
for task_attempt in {1..15}; do
    if dbus-send --session --print-reply --dest=org.freedesktop.DBus \
        /org/freedesktop/DBus org.freedesktop.DBus.NameHasOwner \
        string:org.fcitx.Fcitx5 2>> "$task_log" | grep -q 'boolean true'; then
        task_ready=true
        break
    fi
    sleep 0.2
done
if [[ "$task_ready" != true ]]; then
    echo "输入法未能启动，请查看日志：$task_log"
    exit 1
fi
fcitx5-remote -s pinyin >> "$task_log" 2>&1
fcitx5-remote -o >> "$task_log" 2>&1

echo '正在启动独立的中文输入浏览器窗口。'
echo '点击网页输入框，按 Ctrl+空格或左 Shift 切换拼音，输入 nihao，按空格选词。'
echo '网页服务需保持运行；若提示无法访问，请先运行 bash start-web.sh。'
echo "日志：$task_log"
# A separate profile prevents Chromium from forwarding this launch to an old
# browser process that inherited a different input method environment.
exec chromium --user-data-dir="$task_root/data/chromium-chinese-profile" \
    --ozone-platform=x11 --no-first-run --no-default-browser-check \
    http://127.0.0.1:8080/ >> "$task_log" 2>&1

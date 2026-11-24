#!/usr/bin/env python3
"""GTK desktop shell with native actions and an embedded local WebKit workspace."""
import json
import logging
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit
from desktop_runtime import APP_ID, ROOT, Backend, local_json


def main():
    import gi
    gi.require_version('Gtk', '3.0')
    gi.require_version('GdkPixbuf', '2.0')
    gi.require_version('WebKit2', '4.1')
    from gi.repository import GdkPixbuf, Gio, GLib, Gtk, WebKit2
    if '--check' in sys.argv:
        print(json.dumps({'gtk': True, 'webkit': True, 'display_available': Gtk.init_check()[0],
                          'root': str(ROOT)}, ensure_ascii=False))
        return 0
    if not Gtk.init_check()[0]:
        print('无法连接图形桌面。请在板卡桌面双击“应急先行”，或在桌面终端运行 ./start-desktop.sh。',
              file=sys.stderr)
        return 1

    class Application(Gtk.Application):
        def __init__(self):
            super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
            self.backend = Backend()
            self.window = self.view = None
            self.state = {}
            self.seen = None
            self.closing = self.fullscreen_on = False
            self.poll_stop = threading.Event()
            self.readers = []

        def do_startup(self):
            Gtk.Application.do_startup(self)
            for name, callback in [
                ('show', lambda *_: self.activate()), ('quit', self.close),
                ('reload', self.reload), ('fullscreen', self.fullscreen),
                ('reports', lambda *_: self.open_folder(ROOT / 'data')),
                ('videos', lambda *_: self.open_folder(ROOT / 'samples/monitor-suite/videos')),
                ('logs', lambda *_: self.open_folder(ROOT / 'data/desktop')),
                ('about', self.about),
            ]:
                action = Gio.SimpleAction.new(name, None)
                action.connect('activate', callback)
                self.add_action(action)
            toggle = Gio.SimpleAction.new_stateful('notifications', None, GLib.Variant('b', True))
            toggle.connect('activate', lambda action, _: action.set_state(
                GLib.Variant('b', not action.get_state().get_boolean())))
            self.add_action(toggle)
            self.set_accels_for_action('app.quit', ['<Primary>q'])
            self.set_accels_for_action('app.reload', ['<Primary>r', 'F5'])
            self.set_accels_for_action('app.fullscreen', ['F11'])

        def do_activate(self):
            if self.window:
                self.window.present()
                return
            self.window = Gtk.ApplicationWindow(application=self, title='应急先行 · 离线监测工作站')
            self.window.set_default_size(1360, 900)
            self.window.set_size_request(900, 600)
            icon_path = str(ROOT / 'desktop/app-icon.svg')
            self.window.set_icon_from_file(icon_path)
            self.window.connect('delete-event', self.close)
            header = Gtk.HeaderBar(title='应急先行', subtitle='离线监测工作站')
            header.set_show_close_button(True)
            self.window.set_titlebar(header)
            refresh = Gtk.Button.new_from_icon_name('view-refresh-symbolic', Gtk.IconSize.BUTTON)
            refresh.set_tooltip_text('重新加载工作页（F5）')
            refresh.connect('clicked', self.reload)
            header.pack_start(refresh)
            menu = Gio.Menu()
            for text, action in [('报告与证据', 'reports'), ('演示视频', 'videos'), ('运行日志', 'logs'),
                                 ('桌面事件通知', 'notifications'), ('全屏 / 退出全屏', 'fullscreen'),
                                 ('关于应急先行', 'about'), ('退出应用', 'quit')]:
                menu.append(text, 'app.' + action)
            button = Gtk.MenuButton()
            button.set_image(Gtk.Image.new_from_icon_name('open-menu-symbolic', Gtk.IconSize.BUTTON))
            button.set_menu_model(menu)
            header.pack_end(button)
            layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            self.stack = Gtk.Stack()
            splash = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
            splash.set_halign(Gtk.Align.CENTER)
            splash.set_valign(Gtk.Align.CENTER)
            icon = GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, 72, 72, True)
            splash.pack_start(Gtk.Image.new_from_pixbuf(icon), False, False, 0)
            self.message = Gtk.Label(label='正在启动本地推理服务…')
            self.message.set_line_wrap(True)
            self.message.set_justify(Gtk.Justification.CENTER)
            splash.pack_start(self.message, False, False, 0)
            self.stack.add_named(splash, 'loading')
            layout.pack_start(self.stack, True, True, 0)
            self.footer = Gtk.Label(label='本机运行 · 离线推理 · 数据保存在板卡')
            self.footer.set_xalign(.03)
            self.footer.set_margin_top(5)
            self.footer.set_margin_bottom(5)
            layout.pack_end(self.footer, False, False, 0)
            self.window.add(layout)
            self.window.show_all()
            threading.Thread(target=self.start_backend, daemon=True).start()

        def start_backend(self):
            try:
                self.backend.start()
                GLib.idle_add(self.load_workspace)
            except Exception as exc:
                logging.exception('Desktop startup')
                GLib.idle_add(self.error, str(exc))

        def load_workspace(self):
            if self.closing:
                return False
            folder = ROOT / 'data/desktop'
            manager = WebKit2.WebsiteDataManager(base_data_directory=str(folder / 'webkit-data'),
                                                base_cache_directory=str(folder / 'webkit-cache'))
            self.context = WebKit2.WebContext.new_with_website_data_manager(manager)
            self.context.connect('download-started', self.download)
            self.view = WebKit2.WebView.new_with_context(self.context)
            self.view.connect('decide-policy', self.navigation)
            self.view.connect('load-changed', self.loaded)
            self.view.connect('load-failed', self.load_failed)
            self.view.connect('web-process-terminated', lambda *_:
                              self.error('显示进程已停止。监测状态见底栏，按 F5 重新加载。'))
            self.stack.add_named(self.view, 'workspace')
            self.view.show()
            self.view.load_uri(self.backend.url + '/?desktop=1#monitor-panel')
            threading.Thread(target=self.poll, daemon=True).start()
            return False

        def allowed(self, uri):
            target, origin = urlsplit(uri or ''), urlsplit(self.backend.url or '')
            return (target.scheme, target.netloc) == (origin.scheme, origin.netloc)

        def navigation(self, view, decision, kind):
            if kind not in (WebKit2.PolicyDecisionType.NAVIGATION_ACTION,
                            WebKit2.PolicyDecisionType.NEW_WINDOW_ACTION):
                return False
            uri = decision.get_navigation_action().get_request().get_uri()
            if not self.allowed(uri):
                decision.ignore()
                self.footer.set_text('已阻止离开本地工作站的链接')
                return True
            if kind == WebKit2.PolicyDecisionType.NEW_WINDOW_ACTION:
                decision.ignore()
                reader = Gtk.ApplicationWindow(application=self, title='应急先行 · 报告与证据')
                reader.set_default_size(1050, 760)
                content = WebKit2.WebView.new_with_context(self.context)
                content.connect('decide-policy', self.navigation)
                reader.add(content)
                self.readers.append(reader)
                reader.connect('destroy', lambda window: self.readers.remove(window)
                               if window in self.readers else None)
                reader.show_all()
                content.load_uri(uri)
                return True
            return False

        def loaded(self, view, event):
            if event == WebKit2.LoadEvent.FINISHED:
                self.stack.set_visible_child_name('workspace')

        def load_failed(self, view, event, uri, error):
            self.error('工作页加载失败：' + error.message + '。按 F5 重试。')
            return True

        def error(self, message):
            if not self.closing:
                self.message.set_text(message)
                self.stack.set_visible_child_name('loading')
            return False

        def reload(self, *_):
            if self.view:
                self.view.reload()

        def poll(self):
            while not self.poll_stop.is_set():
                try:
                    GLib.idle_add(self.update, local_json(self.backend.url + '/api/monitor'))
                except (OSError, ValueError) as exc:
                    logging.warning('Monitor status: %s', exc)
                    GLib.idle_add(self.footer.set_text, '本地服务连接中断 · 请查看运行日志')
                if self.poll_stop.wait(2):
                    break

        def update(self, state):
            if self.closing:
                return False
            self.state = state
            names = {'starting': '正在加载模型', 'running': '持续监测中', 'stopping': '正在停止',
                     'stopped': '监测已停止', 'completed': '本轮完成', 'failed': '监测故障'}
            tail = ('退出应用会停止本次本地服务' if self.backend.owned
                    else '已连接现有服务 · 关闭窗口后后台继续运行')
            self.footer.set_text(names.get(state.get('status'), '等待开始监测') + '  ·  ' + tail)
            current = {(state.get('session_id'), event['id']): event.get('stage')
                       for event in state.get('events', [])}
            if self.seen is not None and self.lookup_action('notifications').get_state().get_boolean():
                for event in state.get('events', []):
                    key = (state.get('session_id'), event['id'])
                    if event.get('status') != 'active' or self.seen.get(key) == event.get('stage'):
                        continue
                    label = {'fire': '明火', 'smoke': '烟雾', 'person': '人员'}.get(event['label'], '目标')
                    stage = '连续检出' if event.get('stage') == 'persistent' else '首次检出'
                    notice = Gio.Notification.new('应急先行 · ' + label + stage)
                    notice.set_body('请查看检测画面并核查现场。')
                    notice.set_default_action('app.show')
                    if event['label'] == 'fire':
                        notice.set_priority(Gio.NotificationPriority.URGENT)
                    self.send_notification(event['id'], notice)
            self.seen = current
            return False

        def download(self, context, item):
            if not self.allowed(item.get_request().get_uri()):
                item.cancel()
                return
            def choose(download, suggested):
                dialog = Gtk.FileChooserDialog(title='保存报告或证据', transient_for=self.window,
                                                action=Gtk.FileChooserAction.SAVE)
                dialog.add_buttons('取消', Gtk.ResponseType.CANCEL, '保存', Gtk.ResponseType.ACCEPT)
                dialog.set_do_overwrite_confirmation(True)
                dialog.set_current_name(Path(suggested).name or 'report')
                if dialog.run() == Gtk.ResponseType.ACCEPT:
                    download.set_allow_overwrite(True)
                    download.set_destination(Path(dialog.get_filename()).as_uri())
                else:
                    download.cancel()
                dialog.destroy()
                return True
            item.connect('decide-destination', choose)
            item.connect('failed', lambda _, error: logging.info('Download: %s', error.message))

        def open_folder(self, folder):
            folder.mkdir(parents=True, exist_ok=True)
            try:
                Gio.AppInfo.launch_default_for_uri(folder.as_uri(), None)
            except GLib.Error as exc:
                self.error('无法打开文件夹：' + exc.message)

        def fullscreen(self, *_):
            self.fullscreen_on = not self.fullscreen_on
            self.window.fullscreen() if self.fullscreen_on else self.window.unfullscreen()

        def about(self, *_):
            dialog = Gtk.AboutDialog(transient_for=self.window, modal=True,
                program_name='应急先行', version='0.1 · 桌面预览版',
                comments='离线视频监测、现场分析与证据管理。', logo_icon_name=APP_ID)
            dialog.run()
            dialog.destroy()

        def close(self, *_):
            if self.closing:
                return True
            if self.backend.owned and self.backend.url:
                try:
                    self.state = local_json(self.backend.url + '/api/monitor', timeout=.5)
                except (OSError, ValueError):
                    pass
            active = self.state.get('status') in {'starting', 'running', 'stopping'}
            if self.backend.owned and (active or self.backend.has_active_jobs()):
                dialog = Gtk.MessageDialog(transient_for=self.window, modal=True,
                    message_type=Gtk.MessageType.QUESTION, buttons=Gtk.ButtonsType.NONE,
                    text='停止当前任务并退出应急先行？')
                dialog.format_secondary_text('正在执行的检测和分析会停止，已保存的记录与证据保留。')
                dialog.add_buttons('继续运行', Gtk.ResponseType.CANCEL, '停止并退出', Gtk.ResponseType.ACCEPT)
                response = dialog.run()
                dialog.destroy()
                if response != Gtk.ResponseType.ACCEPT:
                    return True
            self.closing = True
            self.poll_stop.set()
            self.window.set_sensitive(False)
            self.footer.set_text('正在关闭本地服务…')
            self.hold()
            threading.Thread(target=self.close_backend, daemon=True).start()
            return True

        def close_backend(self):
            try:
                self.backend.stop()
            finally:
                GLib.idle_add(self.finish_close)

        def finish_close(self):
            for window in list(self.readers):
                window.destroy()
            self.window.destroy()
            self.release()
            self.quit()
            return False

    folder = ROOT / 'data/desktop'
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=folder / 'desktop.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    app = Application()
    try:
        return app.run([sys.argv[0]])
    finally:
        app.poll_stop.set()
        app.backend.stop()


if __name__ == '__main__':
    raise SystemExit(main())

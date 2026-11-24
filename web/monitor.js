/* Video monitoring uses only same-origin endpoints and local browser audio. */
(() => {
  const get = id => document.getElementById(id);
  let state = {status: 'idle', events: [], sources: []};
  let busy = false;
  let uploading = false;
  let timer;
  let audio;
  let soundEnabled = false;
  let initial = true;
  let lastFrame = '';
  let connectionError = false;
  const notices = new Map();
  const activeStates = ['starting', 'running', 'stopping'];
  const labels = {fire: '明火候选', smoke: '烟雾候选', person: '人员出现'};
  const statusNames = {idle: '尚未启动', starting: '正在加载模型', running: '持续检测中', stopped: '已停止', completed: '回放结束', failed: '监测故障'};
  const reportError = message => {get('monitor-error').textContent = message; get('monitor-error').hidden = !message;};
  const make = (tag, text, className) => {const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element;};
  async function request(path, body) {
    const response = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '监测请求失败');
    return data;
  }
  function sourceNote() {
    const source = state.sources.find(item => item.id === get('monitor-source').value);
    get('monitor-source-note').textContent = source ? `${source.kind === 'image_sequence_demo' ? '模拟演示：静态测试图片拼接，不代表真实监控效果。' : '本地视频回放。'}时长 ${source.duration_seconds} 秒 · ${source.width} × ${source.height}` : '先导入一个视频，再开始监测。';
  }
  function controls() {
    const active = activeStates.includes(state.status);
    for (const id of ['monitor-source', 'monitor-interval', 'monitor-loop', 'monitor-people', 'monitor-person-alert']) get(id).disabled = active || busy;
    get('monitor-person-alert').disabled ||= !get('monitor-people').checked;
    get('monitor-video').disabled = busy || uploading;
    get('monitor-start').disabled = active || busy || uploading || !get('monitor-source').value;
    get('monitor-stop').disabled = !active || busy;
    get('monitor-start').textContent = state.status === 'starting' ? '正在加载本地模型…' : state.status === 'running' ? '正在持续监测…' : '开始持续监测 →';
  }
  function boxes(detections, size) {
    const svg = get('monitor-boxes');
    svg.replaceChildren();
    if (!size) return;
    svg.setAttribute('viewBox', `0 0 ${size[0]} ${size[1]}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const namespace = 'http://www.w3.org/2000/svg';
    for (const detection of detections || []) {
      if (!detection.box) continue;
      const [x1,y1,x2,y2] = detection.box;
      const color = detection.label === 'fire' ? '#ff8069' : detection.label === 'smoke' ? '#ffd269' : '#56ebaa';
      const rect = document.createElementNS(namespace, 'rect');
      for (const [name,value] of Object.entries({x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:color,'stroke-width':3})) rect.setAttribute(name,value);
      const text = document.createElementNS(namespace, 'text');
      text.setAttribute('x', x1); text.setAttribute('y', Math.max(22, y1-6)); text.setAttribute('font-size', 20); text.setAttribute('fill', color); text.setAttribute('stroke', '#163e35'); text.setAttribute('stroke-width', .4);
      text.textContent = `${labels[detection.label] || detection.label} ${Math.round(detection.confidence*100)}%`;
      svg.append(rect,text);
    }
  }
  async function beep() {
    if (!soundEnabled || !audio) return;
    try {
      await audio.resume();
      if (audio.state !== 'running') {soundEnabled = false; get('monitor-sound').textContent = '声音已暂停，点击重新启用'; get('monitor-sound').setAttribute('aria-pressed', 'false'); return;}
      for (const offset of [0,.35]) {
        const oscillator = audio.createOscillator(), gain = audio.createGain();
        oscillator.type = 'sine'; oscillator.frequency.value = 880;
        gain.gain.setValueAtTime(0, audio.currentTime + offset);
        gain.gain.linearRampToValueAtTime(.12, audio.currentTime + offset + .02);
        gain.gain.linearRampToValueAtTime(0, audio.currentTime + offset + .22);
        oscillator.connect(gain); gain.connect(audio.destination);
        oscillator.start(audio.currentTime + offset); oscillator.stop(audio.currentTime + offset + .24);
      }
    } catch (error) {reportError('页面声音不可用：' + error.message);}
  }
  function render(next) {
    state = next;
    get('monitor-state').textContent = statusNames[state.status] || state.status;
    get('monitor-state').dataset.state = state.status;
    if (state.frame_url !== lastFrame && state.frame_url) {
      lastFrame = state.frame_url;
      const frame = get('monitor-frame');
      const snapshot = state;
      frame.onload = () => {get('monitor-empty').hidden = true; frame.hidden = false; boxes(snapshot.detections, snapshot.image_size);};
      frame.onerror = () => {frame.hidden = true; get('monitor-empty').hidden = false; get('monitor-boxes').replaceChildren();};
      frame.src = state.frame_url;
    }
    const active = (state.events || []).filter(event => event.status === 'active');
    const banner = get('monitor-banner');
    banner.dataset.level = active.some(e=>e.level==='critical') ? 'critical' : active.some(e=>e.label!=='person') ? 'warning' : active.length ? 'notice' : '';
    banner.textContent = state.status === 'failed' ? '监测故障 · 当前无法继续检测' : !activeStates.includes(state.status) ? (state.message || '尚未开始监测') : active.length ? active.map(event=>`${labels[event.label]} · ${event.stage === 'persistent' ? '连续检出' : '首次检出'}`).join(' / ') : state.status === 'starting' ? '首次启动需要加载本地模型，请稍候…' : '持续检测中 · 当前未检出提醒目标';
    get('monitor-stats').replaceChildren();
    if (state.frames_processed) {
      const values = [`已检测 ${state.frames_processed} 帧`, `视频位置 ${state.position_seconds} 秒`, `本轮耗时 ${state.cycle_seconds} 秒`, state.people_count === null ? '人员检测关闭' : `本帧人员 ${state.people_count}`];
      for (const value of values) get('monitor-stats').append(make('span',value));
      if (state.status !== 'running') get('monitor-stats').append(make('span','画面为最后完成检测的帧'));
    }
    const list = get('monitor-event-list'); list.replaceChildren();
    get('monitor-event-empty').hidden = !!state.events?.length;
    let notify = false;
    for (const event of state.events || []) {
      const previous = notices.get(event.id);
      if (!initial && event.status === 'active' && previous !== event.stage) notify = true;
      notices.set(event.id,event.stage);
      const row = make('a',undefined,'monitor-event'); row.href = event.image_url; row.target = '_blank'; row.rel = 'noopener'; row.dataset.level = event.level;
      row.append(make('strong',labels[event.label]),make('span',`${event.status === 'ended' ? '已结束' : event.stage === 'persistent' ? '连续检出' : '首次检出'} · ${Math.round(event.confidence*100)}%`),make('small',`${new Date(event.first_seen).toLocaleTimeString('zh-CN')} · 视频 ${event.first_position_seconds} 秒起 · ${event.samples} 次检出 · 查看截图 ↗`));
      list.append(row);
    }
    initial = false;
    if (notify) beep();
    while (notices.size > 400) notices.delete(notices.keys().next().value);
    const select = get('monitor-source'), selected = select.value;
    select.replaceChildren(...(state.sources || []).map(item => {const option = make('option',item.name); option.value = item.id; return option;}));
    if (state.sources?.length) select.value = activeStates.includes(state.status) ? state.source.id : state.sources.some(item=>item.id===selected) ? selected : state.sources[0].id;
    else {const option = make('option','请先导入本地视频'); option.value = ''; select.append(option);}
    if (activeStates.includes(state.status)) {
      get('monitor-interval').value = String(state.interval_seconds);
      get('monitor-loop').checked = state.loop;
      get('monitor-people').checked = state.detect_people;
      get('monitor-person-alert').checked = state.person_alert;
    }
    sourceNote(); controls();
    if (state.error) reportError(state.error);
  }
  async function refresh() {
    clearTimeout(timer);
    try {render(await request('/api/monitor')); if (connectionError && !state.error) reportError(''); connectionError = false;}
    catch (error) {connectionError = true; get('monitor-state').textContent = '连接中断'; get('monitor-state').dataset.state = 'failed'; get('monitor-banner').textContent = '无法确认后台状态 · 请检查网页服务连接'; reportError(error.message);}
    finally {timer = setTimeout(refresh,2000);}
  }
  get('monitor-source').onchange = () => {sourceNote(); controls();};
  get('monitor-people').onchange = controls;
  get('monitor-start').onclick = async () => {
    if (busy) return;
    busy = true; controls(); reportError('');
    try {
      initial = false;
      const next = await request('/api/monitor/start',{source_id:get('monitor-source').value, interval_seconds:Number(get('monitor-interval').value), loop:get('monitor-loop').checked, detect_people:get('monitor-people').checked, person_alert:get('monitor-person-alert').checked});
      next.sources = state.sources; render(next);
    } catch (error) {reportError(error.message);}
    finally {busy = false; controls();}
  };
  get('monitor-stop').onclick = async () => {
    if (busy) return;
    busy = true; controls();
    try {render(await request('/api/monitor/stop',{}));} catch (error) {reportError(error.message);}
    finally {busy = false; controls();}
  };
  get('monitor-video').onchange = async () => {
    const file = get('monitor-video').files[0];
    if (!file) return;
    if (file.size > 200*1024*1024 || !file.size) {reportError('视频不能为空且不能超过 200 MB'); return;}
    uploading = true; controls(); reportError('');
    get('monitor-source-note').textContent = '正在导入到板卡并检查视频，请稍候…';
    try {
      const response = await fetch('/api/monitor/video',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-File-Name':encodeURIComponent(file.name)},body:file});
      const source = await response.json(); if (!response.ok) throw new Error(source.error || '视频导入失败');
      render(await request('/api/monitor'));
      if (!activeStates.includes(state.status)) get('monitor-source').value = source.id;
      sourceNote();
    } catch (error) {reportError(error.message);}
    finally {uploading = false; get('monitor-video').value = ''; controls();}
  };
  get('monitor-sound').onclick = async () => {
    try {
      if (!soundEnabled) {const Audio = window.AudioContext || window.webkitAudioContext; if (!Audio) throw new Error('浏览器不支持音频'); audio ||= new Audio(); await audio.resume(); if (audio.state !== 'running') throw new Error('请允许浏览器播放声音');}
      soundEnabled = !soundEnabled;
      get('monitor-sound').textContent = soundEnabled ? '页面提示音已启用 · 点击静音' : '启用页面提示音';
      get('monitor-sound').setAttribute('aria-pressed',String(soundEnabled));
      if (soundEnabled) beep();
    } catch (error) {reportError('无法启用提示音：' + error.message);}
  };
  refresh();
})();

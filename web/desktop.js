/* Desktop navigation preserves form state and the monitoring controller. */
(() => {
  if (new URLSearchParams(location.search).get('desktop') !== '1') return;
  document.documentElement.dataset.desktop = 'true';
  const pages = ['monitor-panel', 'workspace', 'history-panel', 'batch-panel', 'knowledge-panel'];
  const titles = {
    'monitor-panel': ['持续监测', '选择视频，自动抽帧，记录人员与烟火线索。'],
    workspace: ['现场分析', '本地图片检测、应急问答与证据复核。'],
    'history-panel': ['巡查记录', '回看现场分析，继续对话，保存与导出报告。'],
    'batch-panel': ['批量测试', '查看本地图片集的检测结果与复核记录。'],
    'knowledge-panel': ['本地知识库', '导入和维护离线应急参考资料。']
  };
  function show(id) {
    if (!pages.includes(id)) id = 'monitor-panel';
    for (const key of pages) {
      const element = document.getElementById(key);
      element.hidden = key !== id;
      if (key === id && key === 'knowledge-panel') element.open = true;
    }
    for (const link of document.querySelectorAll('.nav-link')) {
      if (link.hash === '#' + id) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    document.querySelector('.heading h1').textContent = titles[id][0];
    document.querySelector('.heading p').textContent = titles[id][1];
    document.querySelector('.breadcrumb strong').textContent = titles[id][0];
    document.querySelector('main').scrollTop = 0;
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('.nav-link, .brand, .skip-link');
    if (!link) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const id = link.hash.slice(1) || 'monitor-panel';
    location.hash = id;
    show(id);
  }, true);
  addEventListener('hashchange', () => show(location.hash.slice(1)));
  window.desktopNavigate = id => {
    location.hash = pages.includes(id) ? id : 'monitor-panel';
    show(id);
  };
  document.querySelector('.brand small').textContent = '离线监测工作站';
  document.querySelector('.monitor-intro').textContent =
    '选择板端视频；拼接照片素材只用于流程演示。检测到人员或烟火候选后，保存事件帧供复核。';
  const hints = document.querySelectorAll('.monitor-settings > p.hint');
  hints[hints.length - 1].textContent =
    '切换工作页和最小化后监测继续。提示音需点击启用；退出应用前请查看底栏的服务状态。';
  show(location.hash.slice(1));
})();

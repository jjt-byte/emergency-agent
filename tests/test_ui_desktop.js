// Test visible page transitions, preservation of entered data, and browser isolation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('web/desktop.js', 'utf8');
function setup(search) {
  const ids = ['monitor-panel', 'workspace', 'history-panel', 'batch-panel', 'knowledge-panel'];
  const elements = new Map(ids.map(id => [id, {hidden: false, entered: 'kept', open: false}]));
  const links = ids.map(id => ({hash: '#' + id, setAttribute(k, v) {this[k] = v;},
    removeAttribute(k) {delete this[k];}}));
  const headings = new Map(['.heading h1', '.heading p', '.breadcrumb strong',
    'main', '.brand small', '.monitor-intro'].map(key => [key, {textContent: '', scrollTop: 99}]));
  const hints = [{textContent: 'source note'}, {textContent: 'old sound hint'}];
  const listeners = {};
  const document = {documentElement: {dataset: {}},
    getElementById: id => elements.get(id),
    querySelector: selector => headings.get(selector),
    querySelectorAll: selector => selector === '.nav-link' ? links : hints,
    addEventListener: (name, callback) => {listeners[name] = callback;}};
  const location = {search, hash: '#monitor-panel', pathname: '/'};
  // The application declares history() globally; navigation must not call it as a browser object.
  const context = {document, location, window: {}, URLSearchParams, history() {},
    addEventListener: (name, callback) => {listeners[name] = callback;}};
  vm.runInNewContext(source, context);
  return {elements, links, headings, hints, listeners, document, location, context};
}
const desktop = setup('?desktop=1');
assert.equal(desktop.elements.get('monitor-panel').hidden, false);
assert.equal(desktop.elements.get('workspace').hidden, true);
let prevented = false;
desktop.listeners.click({target: {closest: () => desktop.links[1]},
  preventDefault() {prevented = true;}, stopImmediatePropagation() {}});
assert.equal(prevented, true);
assert.equal(desktop.elements.get('workspace').hidden, false);
assert.equal(desktop.elements.get('monitor-panel').hidden, true);
assert.equal(desktop.elements.get('workspace').entered, 'kept');
assert.equal(desktop.headings.get('.heading h1').textContent, '现场分析');
desktop.context.window.desktopNavigate('knowledge-panel');
assert.equal(desktop.elements.get('knowledge-panel').hidden, false);
assert.equal(desktop.elements.get('knowledge-panel').open, true);
assert.equal(desktop.hints[0].textContent, 'source note');
desktop.context.window.desktopNavigate('unknown');
assert.equal(desktop.elements.get('monitor-panel').hidden, false);
const browser = setup('');
assert.deepEqual(browser.document.documentElement.dataset, {});
assert.equal(browser.listeners.click, undefined);
assert.equal(browser.context.window.desktopNavigate, undefined);
console.log('Desktop UI passed: page transitions, retained inputs, record navigation and browser isolation.');

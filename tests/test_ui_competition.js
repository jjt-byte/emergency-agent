const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
class Element {
  constructor(){this.children=[];this.hidden=false;this.attrs={};this._text='';}
  set textContent(value){this._text=value==null?'':String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent||'').join('');}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this._text='';this.children=items;}
  setAttribute(key,value){this.attrs[key]=value;}
}
const elements=new Map();
const el=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const node=(tag,text,className)=>{const item=new Element();item.tag=tag;item.textContent=text||'';item.className=className||'';return item;};
const source=fs.readFileSync('web/app.js','utf8');
const page=fs.readFileSync('web/index.html','utf8');
assert.doesNotMatch(page,/competition-evidence|demo-guide|5 分钟演示路线/);
assert.match(page,/id="monitor-panel"/);
assert.match(page,/id="status-grid"/);
assert.match(page,/5 分钟 Demo 演讲稿/);
assert.doesNotMatch(source,/showEvaluation|prepareDemo|data-demo/);
const context=vm.createContext({el,node,console});
vm.runInContext(source.slice(source.indexOf('function showAgentSteps'),source.indexOf('async function poll')),context);

context.showAgentSteps([
  {title:'第 1 轮工具决策',status:'ok',detail:'选择工具：烟火检测',source:'qwen'},
  {title:'烟火检测',status:'reused',detail:'沿用同一张图片',source:'qwen'},
  {title:'证据校验',status:'rejected',detail:'人数与检测证据不一致',source:'evidence_guard'},
  {title:'整理回答',status:'ok',detail:'使用工具结论',source:'fallback'}
],{status:'fallback',decisions:[
  {action:'tools',tools:[{name:'detect_fire_smoke'}]},
  {action:'finish',answer:'这段模型草稿不能出现在网页决策摘要。'}
]});
assert.equal(el('agent-decisions').children.length,2);
assert.match(el('agent-decisions').textContent,/调用 烟火检测/);
assert.match(el('agent-decisions').textContent,/提交回答 · 进入证据校验/);
assert.doesNotMatch(el('agent-decisions').textContent,/模型草稿/);
assert.match(el('agent-steps').textContent,/Qwen 选择/);
assert.match(el('agent-steps').textContent,/同图记忆复用/);
assert.match(el('agent-steps').textContent,/证据校验规则/);
assert.match(el('agent-summary').textContent,/兜底/);
console.log('Showcase UI passed: monitoring first, compact live status and auditable Agent traces.');

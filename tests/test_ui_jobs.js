// Failed, cancelled and stale requests must never lock the question controls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('web/app.js', 'utf8');
function setup() {
  const elements = new Map(), storage = new Map(), timers = new Map();
  let sequence = 0;
  const get = id => {
    if(!elements.has(id)) elements.set(id, {style:{},children:[],hidden:false,disabled:false,value:'',
      focus(){this.focused=true;},replaceChildren(...items){this.children=items;},
      append(...items){this.children.push(...items);},setAttribute(){},removeAttribute(){}});
    return elements.get(id);
  };
  const errors = [];
  const context = vm.createContext({console,AbortController,Date,
    document:{getElementById:get,createElement:()=>get('node-'+sequence++),createTextNode:text=>({textContent:text})},
    localStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    setTimeout(callback,delay){const id=++sequence;timers.set(id,{callback,delay});return id;},
    clearTimeout(id){timers.delete(id);}
  });
  vm.runInContext(source.slice(0,source.indexOf("el('image-input').onchange=")),context);
  context.error = message => errors.push(message);
  context.history = async()=>{};
  context.showRecord = record => {get('task-answer').textContent=record.task_answer;get('job-status').textContent='已保存';};
  return {context,get,storage,timers,errors,active:()=>vm.runInContext('activeJob',context)};
}
(async()=>{
  const page=setup(), calls=[];
  page.get('task').value='下一问草稿';
  page.storage.set('pending-job','first');
  page.context.api=async(path)=>{calls.push(path);return path.endsWith('/cancel')?{status:'cancelled'}:{status:'running',elapsed_seconds:42};};
  await page.context.poll('first');
  assert.equal(page.get('cancel-button').hidden,false);
  assert.equal(page.get('run-button').disabled,true);
  assert.match(page.get('job-status').textContent,/42 秒/);
  const staleTimer=[...page.timers.values()][0].callback;
  await page.context.stopCurrentJob();
  assert.equal(page.active(),null);
  assert.equal(page.get('run-button').disabled,false);
  assert.equal(page.get('new-conversation').disabled,false);
  assert.equal(page.get('cancel-button').hidden,true);
  assert.equal(page.get('task').value,'下一问草稿');
  assert.equal(page.storage.has('pending-job'),false);
  const before=calls.length;
  await staleTimer();
  assert.equal(calls.length,before);

  page.context.api=async()=>({status:'failed',error:'历史记录处理失败'});
  await page.context.poll('failure');
  assert.equal(page.get('run-button').disabled,false);
  assert.match(page.errors.at(-1),/历史记录处理失败/);

  let rejectOld;
  page.context.api=()=>new Promise((resolve,reject)=>{rejectOld=reject;});
  const old=page.context.poll('old');
  page.context.releaseJob('old');
  page.context.api=async()=>({status:'running',elapsed_seconds:1});
  await page.context.poll('new');
  rejectOld(new Error('旧请求断开'));
  await old;
  assert.equal(page.active(),'new');
  assert.equal(page.get('run-button').disabled,true);
  page.context.releaseJob('new');

  page.context.api=async()=>({status:'done',record:{task_answer:'已完成回答'}});
  page.context.history=async()=>{throw new Error('列表刷新失败');};
  await page.context.poll('done');
  assert.equal(page.active(),null);
  assert.equal(page.get('job-status').textContent,'已保存');
  assert.equal(page.get('task-answer').textContent,'已完成回答');
  assert.equal(page.get('run-button').disabled,false);

  const offline=setup();
  offline.context.fetch=(path,options)=>new Promise((resolve,reject)=>{
    options.signal.addEventListener('abort',()=>{const error=new Error('aborted');error.name='AbortError';reject(error);});
  });
  const polling=offline.context.poll('offline');
  const timeout=[...offline.timers.values()].find(timer=>timer.delay===20000);
  assert.ok(timeout);
  timeout.callback();
  await polling;
  assert.equal(offline.active(),null);
  assert.equal(offline.get('run-button').disabled,false);
  assert.match(offline.errors.at(-1),/超时/);
  console.log('Question controls passed: cancellation, timeout, worker errors and stale polls.');
})().catch(error=>{console.error(error);process.exitCode=1;});

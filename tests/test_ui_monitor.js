// Exercise monitoring through its public controls and polling, without sockets.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() {this.children=[];this.value='';this.hidden=false;this.disabled=false;this.checked=true;this.dataset={};this.files=[];}
  append(...items) {this.children.push(...items);}
  replaceChildren(...items) {this.children=items;if(items[0]?.value!==undefined)this.value=items[0].value;}
  setAttribute(name,value) {this[name]=value;}
}
const elements=new Map();
for(const match of fs.readFileSync('web/index.html','utf8').matchAll(/\bid="([^"]+)"/g)) elements.set(match[1],new Element());
const get=id=>{assert(elements.has(id),'Missing HTML control '+id);return elements.get(id);};
get('monitor-interval').value='2';
const source={id:'demo',name:'模拟演示',kind:'image_sequence_demo',duration_seconds:32,width:960,height:640};
let state={status:'idle',events:[],sources:[]};
let timer;
const requests=[];
let oscillators=0;
class Audio {
  constructor() {this.state='running';this.currentTime=0;this.destination={};}
  async resume() {}
  createOscillator() {oscillators++;return {frequency:{},connect(){},start(){},stop(){}};}
  createGain() {return {gain:{setValueAtTime(){},linearRampToValueAtTime(){}},connect(){}};}
}
const context=vm.createContext({document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element()},
  window:{AudioContext:Audio},console,Date,Map,
  setTimeout(fn) {timer=fn;return 1;},clearTimeout(){},
  fetch:async(path,options)=>{
    requests.push({path,options});
    if(path==='/api/monitor/start') state={...state,status:'starting',source,session_id:'session',...JSON.parse(options.body)};
    if(path==='/api/monitor/stop') state={...state,status:'stopped'};
    return {ok:true,json:async()=>structuredClone(state)};
  }});
vm.runInContext(fs.readFileSync('web/monitor.js','utf8'),context);
const settle=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  await settle();
  assert.equal(get('monitor-start').disabled,true,'Cannot start without a video');
  state={status:'idle',events:[],sources:[source]};
  await timer();
  assert.equal(get('monitor-start').disabled,false);
  assert.ok(get('monitor-source-note').textContent.includes('静态测试图片'));
  await get('monitor-start').onclick();
  assert.equal(get('monitor-start').disabled,true);
  assert.equal(get('monitor-stop').disabled,false);
  assert.equal(get('monitor-source').disabled,true);
  const started=requests.find(r=>r.path==='/api/monitor/start');
  assert.equal(JSON.parse(started.options.body).interval_seconds,2);
  await get('monitor-sound').onclick();
  await settle();
  assert.equal(oscillators,2,'User can unlock local browser audio');
  const event={id:'event-1',label:'fire',stage:'first_seen',level:'warning',status:'active',confidence:.8,
    image_url:'/local.jpg',first_seen:'2026-11-06T00:00:00Z',first_position_seconds:0,samples:1};
  state={...state,status:'running',events:[event],frames_processed:1,position_seconds:0,cycle_seconds:.9,
    frame_url:'/frame-1.jpg',image_size:[640,480],people_count:0,detections:[{label:'fire',confidence:.8,box:[1,2,30,40]}]};
  await timer(); await settle();
  assert.equal(oscillators,4,'First hit triggers a sound');
  assert.equal(get('monitor-event-list').children.length,1);
  assert.equal(get('monitor-banner').dataset.level,'warning');
  get('monitor-frame').onload();
  assert.equal(get('monitor-boxes').children.length,2);
  await timer(); await settle();
  assert.equal(oscillators,4,'Polling an ongoing event does not replay sounds');
  state.events=[{...event,stage:'persistent',level:'critical',samples:2}];
  await timer(); await settle();
  assert.equal(oscillators,6,'Escalation triggers a second sound');
  assert.equal(get('monitor-banner').dataset.level,'critical');
  assert.equal(get('monitor-event-list').children.length,1,'Escalation preserves one event');
  await get('monitor-stop').onclick();
  assert.equal(get('monitor-stop').disabled,true);
  assert.equal(get('monitor-start').disabled,false);
  state={...state,status:'failed',error:'模型推理失败'};
  await timer();
  assert.equal(get('monitor-error').hidden,false);
  assert.ok(get('monitor-banner').textContent.includes('无法继续检测'));
  console.log('Monitoring UI passed: source selection, start/stop, frame overlays, deduplicated sounds, escalation and failure display.');
})().catch(error=>{console.error(error);process.exitCode=1;});

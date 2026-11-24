// Exercise result transitions without a browser or a listening socket.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() { this.style={}; this.children=[]; this.hidden=false; this.textContent=''; }
  replaceChildren(...children) { this.children=children; }
  append(...children) { this.children.push(...children); }
  setAttribute(name,value) { this[name]=value; }
  removeAttribute(name) { delete this[name]; }
}
const elements=new Map();
const document={
  getElementById(id) { if(!elements.has(id))elements.set(id,new Element()); return elements.get(id); },
  createElement:()=>new Element(), createElementNS:()=>new Element(),
  createTextNode:text=>({textContent:text})
};
const context=vm.createContext({document,localStorage:{removeItem(){}},console});
const source=fs.readFileSync('web/app.js','utf8');
vm.runInContext(source.slice(0,source.indexOf("el('image-input').onchange=")),context);
const get=id=>document.getElementById(id);
const base={case_id:'20261030-000000-aaaaaa',risk_level:'待确认',rule_reason:'缺少证据',
  detections:[],knowledge_refs:[],human_confirmation:'未确认',elapsed_seconds:2};
context.showRecord({...base,task:'这张图有几个人？',task_answer:'本张图片估计有 5 人。',
  image:{path:'/evidence/photo.jpg'},request_plan:{show_people:true,show_risk:false},
  person_statistics:{estimated_count:5,high_confidence_count:4,review_count:1}});
assert.equal(get('count').hidden,false);
assert.equal(get('preview').hidden,false);
const answer='第一句：加油。\n第二句：注意安全。';
context.showRecord({...base,task:'写两句鼓励救援队的话',task_answer:answer,
  llm_explanation:answer,request_plan:{show_people:false,show_risk:false}});
assert.equal(get('task-answer').textContent,answer);
assert.equal(get('question').textContent,'本次问题：写两句鼓励救援队的话');
assert.equal(get('count').hidden,true);
assert.equal(get('count-detail').hidden,true);
assert.equal(get('risk-section').hidden,true);
assert.equal(get('image-stage').hidden,true);
assert.equal(get('preview').hidden,true);
assert.equal(get('preview').src,undefined);
assert.equal(get('draft-box').hidden,true);
context.showRecord({...base,task:'这个情况危险吗，应该怎么办？',task_answer:'检出 1 处明火候选、1 处烟雾候选。',
  llm_explanation:'如现场确有烟火，请远离烟火并撤离。',
  request_plan:{show_people:false,show_risk:true,asks_advice:true},image:{path:'/evidence/fire.jpg'},
  vision_tools:{person:{status:'skipped'},fire_smoke:{status:'ok'}},
  vision_result:{image_size:[640,427]},vision_capabilities:['fire','smoke'],
  vision_missing_capabilities:['person','blocked_exit'],detections:[
    {label:'fire',confidence:.95,box:[10,20,100,180]},
    {label:'smoke',confidence:.8,box:[0,0,120,190]}
  ]});
assert.equal(get('count').hidden,true);
assert.equal(get('risk-section').hidden,false);
assert.equal(get('tool-status').textContent,'人员检测：未执行 · 烟火检测：已完成');
assert.equal(get('draft-box').hidden,false);
assert.equal(get('draft-box').open,false);
assert.equal(get('result-details').open,false);
assert.equal(get('boxes').children[0].stroke,'#ff7058');
assert.equal(get('boxes').children[1].textContent,'明火 95%');
assert.equal(get('boxes').children[2].stroke,'#ffd065');
assert.equal(get('boxes').children[3].textContent,'烟雾 80%');
assert.ok(!get('limitations').textContent.includes('本次未覆盖：烟雾'));
assert.ok(!get('limitations').textContent.includes('通道阻塞'));
context.showRecord({...base,task:'检查烟火',task_answer:'烟火工具失败',
  request_plan:{show_risk:true},image:{path:'/evidence/fire.jpg'},
  vision_tools:{fire_smoke:{status:'failed'}},llm_explanation:'补充'});
assert.equal(get('tool-status').textContent,'烟火检测：失败');
assert.equal(get('draft-box').open,false);
context.showRecord({...base,task:'介绍离线问答',task_answer:'无需联网。',request_plan:{}});
assert.equal(get('tool-status').hidden,true);
context.showRecord({...base,task:'检查火情',task_answer:'图片预警0分，不能证明安全。',
  request_plan:{show_risk:true},response_assessment:{score:0,level:0,label:'未检出或线索不足',
    notice:'不是安全保证',actions:['安全位置人工复核'],extinguisher_advice:'未批准自行灭火',
    emergency_contact:'119',basis:['没有检出'],material_guidance:['油锅不用水']}});
assert.equal(get('alert-section').hidden,false);
assert.equal(get('alert-score').textContent,'0 / 100 · 未检出或线索不足');
assert.equal(get('alert-actions').children[0].textContent,'安全位置人工复核');
context.showRecord({...base,task:'这是什么风险',task_answer:'紧急预警，请优先撤离并报警。',
  risk_level:'紧急优先级预警',rule_reason:'图片预警90分，优先撤离并报警。',
  request_plan:{show_risk:true},response_assessment:{score:90,level:3,label:'紧急优先级预警',
    notice:'现场条件需核实，避险不等待核实',actions:['优先撤离并报警'],
    extinguisher_advice:'优先撤离',emergency_contact:'119',basis:[],material_guidance:[]}});
assert.equal(get('risk').textContent,'紧急优先级预警');
assert.equal(get('risk').style.color,'#a53522');
assert.equal(get('alert-score').textContent,'90 / 100 · 紧急优先级预警');
assert.equal(get('confirm-state').textContent,'尚未人工确认');
context.showRecord({...base,task:'纯文字问题',task_answer:'文字回答',request_plan:{}});
assert.equal(get('alert-section').hidden,true);
context.showRecord({...base,task:'那现在该怎么办？',task_answer:'请沿安全路线撤离。',
  conversation_id:'a'.repeat(32),dialogue_history:[{question:'有火吗？',answer:'检出明火候选。'}],
  request_plan:{show_risk:false},agent_execution:{status:'completed'},agent_trace:[
    {title:'烟火检测',status:'reused',detail:'同一图片',source:'qwen'},
    {title:'本地知识检索',status:'ok',detail:'找到3条',source:'qwen'}]});
assert.equal(get('agent-activity').hidden,false);
assert.equal(get('dialogue-box').hidden,false);
assert.equal(get('dialogue-history').children.length,1);
assert.ok(get('agent-summary').textContent.includes('烟火检测（沿用）'));
assert.ok(get('conversation-state').textContent.includes('继续追问'));
get('condition-trained').checked=true;get('fire-type').value='ordinary';
context.resetFieldConfirmations();
assert.equal(get('condition-trained').checked,false);
assert.equal(get('fire-type').value,'unknown');
console.log('UI transitions passed: free text, person counts and smoke/fire evidence.');

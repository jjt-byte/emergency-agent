const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
class Element {
  constructor(){this.children=[];this.hidden=false;this.value='';this.style={};}
  replaceChildren(...items){this.children=items;}
  append(...items){this.children.push(...items);}
}
const elements=new Map();
const el=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const result={status:{id:'20261030-120000',status:'done',completed:3,total:3,errors:1,evaluated:2,
  metrics:{val:{fire:{precision:.5,recall:.25,f1:1/3,tp:1,fp:1,fn:3},smoke:{precision:null,recall:null,f1:null,tp:0,fp:0,fn:0}}}},
  matched:3,items:[{index:0,name:'one.jpg',score:83,label:'紧急优先级预警',case_id:'case1',split:'val'},
                 {index:1,name:'two.jpg',score:0,label:'未检出或线索不足',case_id:'case2',split:'train'},
                 {index:2,name:'bad.jpg',score:null,label:'未评分',error:'图片损坏'}]};
const context=vm.createContext({el,names:{fire:'明火',smoke:'烟雾'},clearTimeout(){},setTimeout(){},
  node(tag,text,cls){const e=new Element();e.textContent=text;e.className=cls;return e;},
  api:async()=>result,error(message){throw Error(message);},console});
const source=fs.readFileSync('web/app.js','utf8');
vm.runInContext(source.slice(source.indexOf('let batchOffset='),source.indexOf("el('batch-start').onclick=")),context);
el('batch-select').value='20261030-120000';
(async()=>{
  await context.showBatch();
  assert.equal(el('batch-export').hidden,false);
  assert.equal(el('batch-export').href,'/api/batches/20261030-120000/export');
  assert.ok(el('batch-status').textContent.includes('工具错误 1'));
  assert.equal(el('batch-results').children.length,3);
  assert.equal(el('batch-results').children[2].disabled,true);
  assert.equal(el('batch-prev').disabled,true);
  assert.equal(el('batch-next').disabled,true);
  assert.equal(el('batch-metrics').children[0].children[1].children[1].textContent,'50.0%');
  assert.equal(el('batch-metrics').children[0].children[2].children[1].textContent,'—');
  console.log('Batch UI passed: progress, metrics, unknown results and CSV export.');
})().catch(error=>{console.error(error);process.exitCode=1;});

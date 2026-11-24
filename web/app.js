/* All assets and API requests are served by the board; no CDN or cloud calls. */
const el = id => document.getElementById(id);
let currentRecord = null;
let previewURL = null;
let activeJob = null;
let pollTimer = null;
let pollGeneration = 0;
let submitting = false;
let conversationId = localStorage.getItem?.('agent-conversation') || null;
let clearScene = false;
let serverAgentEnabled = null;
const fieldConditions=['small_initial_fire','trained','clear_exit','no_dense_smoke','no_spread','matching_extinguisher','no_special_hazard','alarm_arranged'];
const names = {person:'人员',fire:'明火',smoke:'烟雾',blocked_exit:'通道阻塞'};
async function api(path, body, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(()=>controller.abort(),timeoutMs);
  try {
    const response = await fetch(path, {...(body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),signal:controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '请求失败');
    return data;
  } catch (e) {
    if(e.name==='AbortError')throw new Error('连接本机服务超时，请检查网页服务后重试。');
    throw e;
  } finally { clearTimeout(timer); }
}
function error(message) { el('error').textContent = message; el('error').hidden = !message; }
function node(tag, text, className) { const item=document.createElement(tag);if(text!==undefined)item.textContent=text;if(className)item.className=className;return item; }
async function status() {
  const data=await api('/api/status');
  serverAgentEnabled=!!data.agent_enabled;
  const entries=[['人员检测',data.person_ready??(data.npu_device&&data.person_model),data.person_status|| (data.npu_device?'本地 SSD 模型 · NPU':'当前环境未发现 NPU 设备')],['烟雾 / 明火检测',data.fire_smoke_ready,data.fire_smoke_status||'尚未配置'],['本地语言模型',data.llm_model&&data.llm_runtime,'Qwen · CPU 本地推理'],['离线知识库',data.knowledge_count>0,`${data.knowledge_count} 份文档 · 本机保存`]];
  el('status-grid').replaceChildren();
  for(const [title,ok,text] of entries){const card=node('div',undefined,'status');const heading=node('strong');heading.append(node('i',undefined,ok?'dot ok':'dot'),document.createTextNode(title));card.append(heading,node('small',text),node('span',ok?'就绪':'待检查',ok?'status-indicator ready':'status-indicator'));el('status-grid').append(card);}
}
async function history() {
  const records=await api('/api/history');el('history').replaceChildren();
  if(!records.length)el('history').append(node('div','还没有记录，输入第一个问题开始。','history-empty'));
  for(const r of records){const button=node('button',undefined,'history-row');const title=node('span',r.task);title.append(node('small',r.created_at));button.append(title,node('span',r.risk_level),node('span',r.human_confirmation==='未确认'?'等待人工复核':'已填写复核'));button.onclick=async()=>{try{showRecord(await api('/api/records/'+r.case_id), true);el('image-stage').scrollIntoView({behavior:'smooth',block:'center'});}catch(e){error(e.message);}};el('history').append(button);}
}
function showImage(src) {el('image-stage').hidden=false;el('empty').hidden=true;el('preview').hidden=false;el('preview').src=src;el('boxes').replaceChildren();}
function personSummary(record) {
  if(record.person_statistics)return record.person_statistics;
  const available=!record.tool_error&&['cix_ssd_300_npu','cix_yolov8s_npu','damoyolo_person_cpu','external_json_unverified'].includes(record.detection_source);
  const people=(record.detections||[]).filter(x=>x.label==='person');
  const high=people.filter(x=>x.confidence>=0.7).length;
  return {status:available?'available':'unavailable',estimated_count:available?people.length:null,
    high_confidence_count:available?high:null,review_count:available?people.length-high:null,
    review_threshold:0.7,detection_threshold:record.vision_result?.threshold};
}
function drawBoxes(record) {
  const svg=el('boxes');svg.replaceChildren();
  const size=record.vision_result?.image_size;
  if(!size)return;
  svg.setAttribute('viewBox',`0 0 ${size[0]} ${size[1]}`);svg.setAttribute('preserveAspectRatio','xMidYMid meet');
  const ns='http://www.w3.org/2000/svg';
  let personNumber=0;
  for(const item of record.detections||[]){if(item.label==='person')personNumber++;if(!item.box)continue;const [x1,y1,x2,y2]=item.box;const color=item.label==='fire'?'#ff7058':item.label==='smoke'?'#ffd065':item.confidence<0.7?'#ffbf52':'#28f3a6';const rect=document.createElementNS(ns,'rect');for(const [key,value] of Object.entries({x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:color,'stroke-width':3}))rect.setAttribute(key,value);const text=document.createElementNS(ns,'text');text.setAttribute('x',x1);text.setAttribute('y',Math.max(20,y1-6));text.setAttribute('font-size',18);text.setAttribute('fill',color);text.setAttribute('stroke','#153f30');text.setAttribute('stroke-width',0.35);const number=item.label==='person'?` #${personNumber}`:'';text.textContent=`${names[item.label]||item.label}${number} ${(item.confidence*100).toFixed(0)}%`;svg.append(rect,text);}
}
function showRecord(record, reveal = false) {
  if (reveal && typeof window !== 'undefined') window.desktopNavigate?.('workspace');
  currentRecord=record;localStorage.removeItem('pending-job');
  el('result-details').open=false;
  el('result-details').hidden=false;
  showAgentSteps(record.agent_trace||[],record.agent_execution,record.agent_execution?.decisions||[]);
  el('agent-alert').hidden=true;
  el('dialogue-history').replaceChildren();
  const dialogue=record.dialogue_history||[];
  el('dialogue-box').hidden=!dialogue.length;
  el('dialogue-box').open=false;
  for(const turn of dialogue){const item=node('div',undefined,'dialogue-turn');item.append(node('strong','你：'+turn.question),node('p','助手：'+turn.answer));el('dialogue-history').append(item);}
  if(record.conversation_id){conversationId=record.conversation_id;localStorage.setItem?.('agent-conversation',conversationId);}
  else {conversationId=null;localStorage.removeItem('agent-conversation');}
  clearScene=false;
  el('image-input').value='';
  el('file-name').textContent=record.image?'更换现场图片':'选择现场图片';
  el('clear-image').hidden=!record.image;
  el('new-conversation').disabled=false;
  el('conversation-state').textContent=record.conversation_id?(record.image?'当前对话已保留这张图片，直接输入下一问即可；换场景请上传新图或开始新对话。':'可继续追问，Agent会参考前面的对话；换话题可开始新对话。'):'本地Agent可选择检测工具、检索资料并继续追问。';
  if(record.image)showImage('/data/evidence/'+record.image.path.split('/').pop());
  else {el('preview').hidden=true;el('preview').removeAttribute('src');el('empty').hidden=false;el('boxes').replaceChildren();}
  const plan=record.request_plan||{};
  el('image-stage').hidden=!record.image;
  el('risk-section').hidden=!plan.show_risk;
  el('count').hidden=!plan.show_people;
  el('count-detail').hidden=!plan.show_people;
  el('question').textContent='本次问题：'+record.task+(record.dataset_sample?' · 数据集图片 '+record.dataset_sample.name:'');
  el('preview').onload=()=>drawBoxes(record);drawBoxes(record);
  el('risk').textContent=record.risk_level;
  el('risk').style.color=/紧急|高优先级|高风险/.test(record.risk_level)?'#a53522':'#8b6926';
  el('reason').textContent=record.rule_reason;
  const detected=record.detections||[], stats=personSummary(record);
  el('count').textContent=stats.estimated_count===null?'人数未统计':`估计 ${stats.estimated_count} 个人员目标`;
  el('task-answer').textContent=record.task_answer||'这条历史记录尚未生成针对本问题的回答，请重新提交。';
  const assessment=record.response_assessment;
  el('alert-section').hidden=!assessment||!plan.show_risk;
  if(assessment){
    el('alert-score').textContent=`${assessment.score===null?'未评分':assessment.score+' / 100'} · ${assessment.label}`;
    el('alert-section').setAttribute('data-level',assessment.level);
    el('alert-notice').textContent=assessment.notice;
    el('alert-actions').replaceChildren(...assessment.actions.map(text=>node('li',text)));
    el('extinguisher-advice').textContent=assessment.extinguisher_advice;
    el('emergency-contact').textContent=assessment.emergency_contact;
    el('alert-basis').replaceChildren(...assessment.basis.map(text=>node('li',text)));
    el('material-guidance').replaceChildren(...assessment.material_guidance.map(text=>node('li',text)));
  }
  let detail=stats.estimated_count===null?'人数未知，不记为 0 人。':`其中 ${stats.high_confidence_count} 个框置信度 ≥70%，${stats.review_count} 个框低于 70%。橙色框需重点复核。`;
  if(stats.detection_threshold!=null)detail+=` 检测阈值：${(stats.detection_threshold*100).toFixed(0)}%。`;
  el('count-detail').textContent=detail;
  const tools=record.vision_tools||{};
  const toolNames={person:'人员检测',fire_smoke:'烟火检测'};
  const toolStates={ok:'已完成',failed:'失败',skipped:'未执行'};
  el('tool-status').hidden=!record.image;
  el('tool-status').textContent=Object.entries(tools).map(([key,value])=>`${toolNames[key]||key}：${toolStates[value.status]||'未知'}`).join(' · ');
  let personNumber=0;
  el('detections').replaceChildren(...detected.map(x=>{const number=x.label==='person'?` #${++personNumber}`:'';return node('span',`${names[x.label]||x.label}${number} · ${(x.confidence*100).toFixed(1)}%`,'detection');}));
  el('limitations').hidden=!record.image;
  const missing=(record.vision_missing_capabilities||[]).filter(label=>label!=='blocked_exit').map(label=>names[label]||label);
  el('limitations').textContent=(record.tool_error?`部分检测未完成：${record.tool_error_message||'工具失败'}。成功工具的结果已保留。 `:'')+(missing.length?`本次未覆盖：${missing.join('、')}。 `:'')+'检测结果需人工复核，未检出不能证明现场安全。';
  el('draft-box').hidden=(!record.llm_explanation&&!record.llm_error)||record.llm_explanation===record.task_answer;
  el('draft-box').open=false;
  el('draft').textContent=record.llm_explanation||`本地说明未生成：${record.llm_error}`;
  el('refs').replaceChildren();for(const ref of record.knowledge_refs||[]){el('refs').append(node('p',`${ref.title} · ${ref.review_status==='pending_local_review'?'项目整理，待审核 · ':''}第 ${ref.line} 行：${ref.excerpt}`));}
  if(!(record.knowledge_refs||[]).length)el('refs').append(node('p','未检索到匹配的本地预案，可在下方导入适用文档。','hint'));
  el('report-links').replaceChildren();for(const [ext,label] of [['html','打开证据报告 ↗'],['md','文字报告'],['json','原始 JSON']]){const link=node('a',label);link.href=`/data/reports/${record.case_id}.${ext}`;link.target='_blank';link.rel='noopener';el('report-links').append(link);}
  el('confirmation').value=record.human_confirmation==='未确认'?'':record.human_confirmation;
  el('confirm-form').hidden=false;el('confirm-state').textContent=record.human_confirmation==='未确认'?'尚未人工确认':'已保存人工复核记录';
  el('job-status').textContent=`已保存 · ${record.elapsed_seconds||0} 秒`;
}
function showAgentSteps(steps,execution,decisions=execution?.decisions||[]){
  el('agent-activity').hidden=!steps.length&&!decisions.length;
  const labels={running:'进行中',ok:'完成',reused:'沿用结果',failed:'失败',skipped:'跳过',rejected:'已拦截'};
  const sources={qwen:'Qwen 选择',safety_rule:'即时预警规则',memory_guard:'会话证据规则',evidence_guard:'证据校验规则',fallback:'异常兜底',runtime:'执行器'};
  const toolTitles={detect_people:'人员检测',detect_fire_smoke:'烟火检测',search_knowledge:'本地知识检索',assess_risk:'预警评估'};
  const finished=steps.filter(x=>['ok','reused'].includes(x.status)&&['人员检测','烟火检测','本地知识检索','预警评估'].includes(x.title));
  const titles=[...new Set(finished.map(x=>x.title+(x.status==='reused'?'（沿用）':'')))];
  const completed=execution&&['completed','fallback'].includes(execution.status);
  el('agent-summary').textContent=completed?(execution.status==='fallback'?'本轮部分内容使用工具结果兜底。 ':'本地Agent已完成。 ')+titles.join(' → '):'Agent正在处理：'+(steps.at(-1)?.title||'理解问题');
  el('agent-decisions').replaceChildren(...decisions.map((d,index)=>{
    const item=node('li',undefined,'agent-decision');
    const action=d.action==='tools'?'调用 '+(d.tools||[]).map(t=>toolTitles[t.name]||'未知工具').join(' → '):d.action==='ask'?'请求补充信息':d.action==='finish'?'提交回答 · 进入证据校验':'动作无效';
    item.append(node('span',`第 ${index+1} 次模型返回`,'decision-round'),node('strong',action));return item;
  }));
  if(!decisions.length)el('agent-decisions').append(node('li',completed?'该记录未保存逐轮模型决策，请查看下方实际执行来源。':'等待本地模型返回第一轮工具决策…','decision-empty'));
  el('agent-steps').replaceChildren(...steps.map(x=>{
    const item=node('li',undefined,'agent-step');item.setAttribute('data-status',x.status);
    item.append(node('strong',`${x.title} · ${labels[x.status]||x.status}`),node('span',sources[x.source]||x.source||'执行器','trace-source'));
    if(x.status==='reused')item.append(node('span','同图记忆复用','trace-source memory'));
    if(x.detail)item.append(node('p',x.detail));return item;
  }));
}
function releaseJob(id) {
  if(activeJob!==id)return;
  clearTimeout(pollTimer);pollTimer=null;pollGeneration++;
  activeJob=null;localStorage.removeItem('pending-job');
  el('run-button').disabled=false;el('new-conversation').disabled=false;
  el('cancel-button').hidden=true;el('cancel-button').disabled=false;
}
async function stopCurrentJob() {
  const id=activeJob;if(!id)return;
  el('cancel-button').disabled=true;
  try {
    const state=await api('/api/jobs/'+id+'/cancel',{});
    if(activeJob!==id)return;
    if(state.status==='done'&&state.case_id){
      const record=await api('/api/records/'+state.case_id);
      if(activeJob!==id)return;
      showRecord(record);releaseJob(id);
    }else{
      releaseJob(id);
      el('job-status').textContent=state.status==='failed'?'本轮未完成，可以重新提问':'本轮已停止，可以重新提问';
      el('agent-activity').hidden=true;el('task').focus();
    }
    error('');
  } catch(e) {
    if(activeJob===id){el('cancel-button').disabled=false;error('停止未成功：'+e.message);}
    else error(e.message);
  }
}
async function poll(id, generation) {
  if(generation===undefined){
    clearTimeout(pollTimer);generation=++pollGeneration;activeJob=id;
    el('run-button').disabled=true;el('new-conversation').disabled=true;
    el('cancel-button').hidden=false;el('cancel-button').disabled=false;
  }
  if(activeJob!==id||generation!==pollGeneration)return;
  try{const state=await api('/api/jobs/'+id);if(activeJob!==id)return;
    if(generation!==pollGeneration)return;
    if(state.status==='done'){showRecord(state.record);releaseJob(id);history().catch(e=>error('记录列表刷新失败：'+e.message));return;}
    if(state.status==='failed')throw new Error(state.error||'任务失败');
    if(state.status==='cancelled'){releaseJob(id);el('agent-activity').hidden=true;el('job-status').textContent='本轮已停止，可以重新提问';return;}
    if(!['queued','running'].includes(state.status))throw new Error('任务状态异常，请重新提问。');
    if(state.progress){showAgentSteps(state.progress.steps||[],state.progress.execution,state.progress.decisions||[]);el('agent-alert').textContent=state.progress.alert||'';el('agent-alert').hidden=!state.progress.alert;}
    const seconds=Number.isFinite(state.elapsed_seconds)?state.elapsed_seconds:Math.max(0,Math.floor((Date.now()-Date.parse(state.started_at||state.created_at||new Date().toISOString()))/1000));
    const stage=state.progress?.steps?.at(-1)?.title||'正在本机处理';
    el('job-status').textContent=(state.status==='queued'?'任务排队中':stage)+' · 已等待 '+seconds+' 秒，可停止本轮后重新提问';
    pollTimer=setTimeout(()=>poll(id,generation),1000);
  }catch(e){if(activeJob!==id||generation!==pollGeneration)return;releaseJob(id);el('job-status').textContent='任务未完成，可以重新提问';error(e.message);}
}
function resetFieldConfirmations(){for(const name of fieldConditions)el('condition-'+name).checked=false;el('fire-type').value='unknown';}
el('image-input').onchange=()=>{const file=el('image-input').files[0];if(!file)return;clearScene=false;resetFieldConfirmations();if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(file);showImage(previewURL);el('preview').onload=null;el('file-name').textContent=file.name;el('clear-image').hidden=false;};
el('clear-image').onclick=()=>{clearScene=true;resetFieldConfirmations();el('image-input').value='';el('file-name').textContent='上传现场图片';el('clear-image').hidden=true;if(previewURL){URL.revokeObjectURL(previewURL);previewURL=null;}el('preview').hidden=true;el('preview').removeAttribute('src');el('boxes').replaceChildren();el('image-stage').hidden=false;el('empty').hidden=false;el('conversation-state').textContent='已移除现场图片，下一问只使用文字和对话记录。';};
el('new-conversation').onclick=()=>{if(activeJob)return;conversationId=null;localStorage.removeItem('agent-conversation');currentRecord=null;el('clear-image').onclick();clearScene=false;el('task').value='';el('observation').value='';el('task-answer').textContent='新对话已开始，请输入问题。';el('question').textContent='';for(const id of ['agent-activity','dialogue-box','result-details','risk-section','alert-section','confirm-form'])el(id).hidden=true;el('report-links').replaceChildren();el('conversation-state').textContent='新对话：不会沿用前一场景的图片和检测结果。';};
el('run-form').onsubmit=async event=>{
  event.preventDefault();error('');
  if(activeJob||submitting)return;
  if(el('llm').checked&&serverAgentEnabled===false)return error('当前网页服务尚未加载Agent升级，请重启 start-web.sh 后按 Ctrl+F5 刷新。');
  const file=el('image-input').files[0];
  if(file&&file.size>10*1024*1024)return error('图片超过 10 MB');
  submitting=true;el('run-button').disabled=true;el('new-conversation').disabled=true;el('job-status').textContent='正在提交问题…';
  try{
    const safety_context={fire_type:el('fire-type').value};
    for(const name of fieldConditions)safety_context[name]=el('condition-'+name).checked;
    const request={task:el('task').value,observation:el('observation').value,detect:el('detect').checked,llm:el('llm').checked,safety_context,conversation_id:conversationId,parent_case_id:currentRecord?.case_id,clear_scene:clearScene};
    if(file){request.image_name=file.name;request.image_base64=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(new Error('图片读取失败'));reader.readAsDataURL(file);});}
    const job=await api('/api/run',request);
    if(job.conversation_id){conversationId=job.conversation_id;localStorage.setItem('agent-conversation',conversationId);}
    el('agent-activity').hidden=true;el('new-conversation').disabled=true;
    currentRecord=null;el('confirm-form').hidden=true;localStorage.setItem('pending-job',job.id);poll(job.id);
  }catch(e){el('run-button').disabled=false;el('new-conversation').disabled=false;el('job-status').textContent='问题未提交，请重试';error(e.message);}
  finally{submitting=false;}
};
el('cancel-button').onclick=stopCurrentJob;
el('confirm-form').onsubmit=async event=>{event.preventDefault();if(!currentRecord)return;try{const updated=await api('/api/confirm',{case_id:currentRecord.case_id,text:el('confirmation').value});showRecord(updated);await history();}catch(e){error(e.message);}};
el('knowledge-form').onsubmit=async event=>{event.preventDefault();try{await api('/api/knowledge',{title:el('knowledge-title').value,body:el('knowledge-body').value});el('knowledge-state').textContent='已保存到本机知识库';el('knowledge-body').value='';await status();}catch(e){error(e.message);}};
let batchOffset=0,batchTimer=null;
async function batchList(){
  const items=await api('/api/batches'),select=el('batch-select'),selected=select.value;
  select.replaceChildren(...items.map(item=>{const option=node('option',`${item.id} · ${item.status} · ${item.completed||0}/${item.total||'?'}`);option.value=item.id;return option;}));
  if(items.length){select.value=items.some(x=>x.id===selected)?selected:items[0].id;await showBatch();}
  else {el('batch-status').textContent='还没有批量测试，点击上方按钮开始。';el('batch-prev').disabled=true;el('batch-next').disabled=true;}
}
async function showBatch(){
  clearTimeout(batchTimer);
  const id=el('batch-select').value;if(!id)return;
  const level=el('batch-level').value||'';
  const result=await api(`/api/batches/${id}?offset=${batchOffset}&limit=30&level=${encodeURIComponent(level)}`);
  if(id!==el('batch-select').value)return;
  const s=result.status;
  const states={done:'已完成',running:'正在检测',preparing:'正在解压准备',queued:'排队中',failed:'失败'};
  el('batch-status').textContent=`${states[s.status]||s.status} · ${s.completed||0} / ${s.total||'?'} 张 · 工具错误 ${s.errors||0} · 已评估标注 ${s.evaluated||0} 张${s.annotation_errors?' · 无效标注 '+s.annotation_errors+' 张':''}${s.error?' · '+s.error:''}`;
  el('batch-metrics').replaceChildren();
  const percent=value=>value==null?'—':(value*100).toFixed(1)+'%';
  for(const [scope,classes] of Object.entries(s.metrics||{})){
    const table=node('table'),head=node('tr');
    for(const title of ['范围 / 类别','Precision','Recall','F1','TP / FP / FN'])head.append(node('th',title));table.append(head);
    for(const [label,m] of Object.entries(classes)){const row=node('tr');for(const value of [`${scope==='val'?'数据集val列表':'全部有标注图片'} / ${names[label]||label}`,percent(m.precision),percent(m.recall),percent(m.f1),`${m.tp} / ${m.fp} / ${m.fn}`])row.append(node('td',value));table.append(row);}
    el('batch-metrics').append(table);
  }
  if(s.image_presence_metrics){
    el('batch-metrics').append(node('strong','图片级有无检测：检出任一目标即算该类出现，不要求框位置匹配'));
    for(const [scope,classes] of Object.entries(s.image_presence_metrics)){
      const table=node('table'),head=node('tr');for(const title of ['范围 / 类别','Precision','Recall','F1','TP / FP / FN / TN'])head.append(node('th',title));table.append(head);
      for(const [label,m] of Object.entries(classes)){const row=node('tr');for(const value of [`${scope==='val'?'数据集val列表':'全部有标注图片'} / ${names[label]||label}`,percent(m.precision),percent(m.recall),percent(m.f1),`${m.tp} / ${m.fp} / ${m.fn} / ${m.tn}`])row.append(node('td',value));table.append(row);}el('batch-metrics').append(table);
    }
  }
  el('batch-metrics').append(node('p','Precision（查准率）表示检出框中匹配标注的比例；Recall（召回率）表示标注目标中被检出的比例。它们用于评估模型，与每张图片的预警分含义不同。','hint'));
  el('batch-metrics').append(node('p','固定阈值下的目标框评估；包含数据集训练列表，预训练数据重叠未知，不能当作独立泛化测试或现场风险准确率。','hint'));
  el('batch-metrics').append(node('p',s.annotation_quality_note||'数据集标注质量尚未核验，以上指标按提供的标注计算，需结合图片复核。','hint'));
  el('batch-metrics').append(node('p',s.score_calibration_status||'数据只有烟火目标框，没有真实火势等级标签，四档预警分尚未校准。','hint'));
  el('batch-results').replaceChildren(...result.items.map(item=>{
    const button=node('button',undefined,'batch-result');
    button.append(node('span',item.name),node('strong',item.score===null?'未评分':`${item.score}分 · ${item.label}`),node('small',item.error||item.annotation_error||`${item.split} · 点击查看检测框与应对建议`));
    button.disabled=!item.case_id;
    button.onclick=async()=>{try{showRecord(await api('/api/records/'+item.case_id), true);el('task-answer').scrollIntoView({behavior:'smooth',block:'center'});}catch(e){error(e.message);}};return button;
  }));
  el('batch-page').textContent=`${result.matched?batchOffset+1:0}—${Math.min(batchOffset+30,result.matched)} / ${result.matched}`;
  el('batch-prev').disabled=batchOffset===0;el('batch-next').disabled=batchOffset+30>=result.matched;
  el('batch-export').hidden=s.status!=='done';el('batch-export').href=`/api/batches/${id}/export`;el('batch-export').download='smoke-fire-results.csv';
  el('batch-report').hidden=!s.report_finalized_at;el('batch-report').href=`/data/batches/${id}/summary.html`;el('batch-report').target='_blank';
  el('batch-start').disabled=['queued','preparing','running'].includes(s.status);
  if(['queued','preparing','running'].includes(s.status))batchTimer=setTimeout(()=>showBatch().catch(e=>error(e.message)),5000);
}
el('batch-start').onclick=async()=>{try{el('batch-start').disabled=true;await api('/api/batches',{});batchOffset=0;await batchList();}catch(e){el('batch-start').disabled=false;error(e.message);}};
el('batch-select').onchange=()=>{batchOffset=0;showBatch().catch(e=>error(e.message));};
el('batch-level').onchange=()=>{batchOffset=0;showBatch().catch(e=>error(e.message));};
el('batch-prev').onclick=()=>{batchOffset=Math.max(0,batchOffset-30);showBatch().catch(e=>error(e.message));};
el('batch-next').onclick=()=>{batchOffset+=30;showBatch().catch(e=>error(e.message));};
el('refresh').onclick=()=>Promise.all([status(),history(),batchList()]).catch(e=>error(e.message));
Promise.all([status(),history(),batchList()]).then(async()=>{const pending=localStorage.getItem('pending-job');if(pending){el('run-button').disabled=true;poll(pending);}else if(conversationId&&serverAgentEnabled){try{const session=await api('/api/conversations/'+conversationId);if(session.latest_record)showRecord(session.latest_record);}catch(e){conversationId=null;localStorage.removeItem('agent-conversation');}}}).catch(e=>error(e.message));

for (const button of document.querySelectorAll('[data-prompt]')) {
  button.addEventListener('click', () => {
    if (activeJob || el('run-button').disabled) return;
    el('task').value = button.dataset.prompt;
    el('task').focus();
  });
}
const navigation = [...document.querySelectorAll('.nav-link')];
function highlightNavigation(link) {
  for (const item of navigation) item.removeAttribute('aria-current');
  link.setAttribute('aria-current', 'location');
}
for (const link of navigation) {
  link.addEventListener('click', () => {
    highlightNavigation(link);
    if (link.id === 'nav-knowledge') el('knowledge-panel').open = true;
  });
}
if ('IntersectionObserver' in window && new URLSearchParams(location.search).get('desktop') !== '1') {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const link = navigation.find(item => item.hash === '#' + entry.target.id);
      if (link) highlightNavigation(link);
    }
  }, {rootMargin: '-5% 0px -65% 0px'});
  for (const id of ['monitor-panel', 'workspace', 'history-panel', 'batch-panel', 'knowledge-panel']) observer.observe(el(id));
}

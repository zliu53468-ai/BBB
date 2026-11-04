'use strict';
const assert=require('node:assert/strict'),vm=require('vm'),fs=require('fs');
const core=require('./pure_card_runtime.js');
function mount(){
 const ids=['directionText','directionCode','regime','message','confidence','strength','directionOrb','pointInputStatus','pointPlayerFinal','pointBankerFinal','btnPointAnalyze','bbbPointEntry','modePill','btnBack','localCalibrationStatus','btnEnd'];
 const nodes={},jobs=[],data=new Map();let workerFails=false;
 function element(){return {value:'',textContent:'',disabled:false,hidden:false,style:{},handlers:{},appendChild(){},addEventListener(k,fn){(this.handlers[k]??=[]).push(fn);},click(){for(const f of this.handlers.click||[])f({preventDefault(){},stopImmediatePropagation(){}});}};}
 for(const id of ids)nodes[id]=element();
 const document={getElementById:id=>nodes[id]||null,querySelectorAll:()=>[],querySelector:()=>null,
  createElement:()=>{const e=element();Object.defineProperty(e,'id',{set(id){nodes[id]=e;}});return e;},createTextNode:s=>s};
 class Worker{
  constructor(){if(workerFails)throw Error('load failed');jobs.push(this);}
  postMessage(x){this.input=x;}terminate(){this.terminated=true;}
  respond(report){this.onmessage({data:{id:this.input.id,report}});}
 }
 const root={localStorage:{getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v)},Worker,matchMedia:()=>({matches:false})};
 vm.runInNewContext(fs.readFileSync('pure_card_panel.js','utf8'),{window:root,document,setTimeout:()=>1,clearTimeout(){},console});
 return {root,nodes,jobs,failWorker(){workerFails=true;}};
}
const v=mount();v.nodes.pointPlayerFinal.value='4+4';v.nodes.pointBankerFinal.value='3+4';
v.nodes.btnPointAnalyze.click();assert.equal(v.nodes.directionText.textContent,'觀望');
const good=core.run([4,4],[3,4]);v.jobs[0].respond(good);
assert.equal(v.root.__BBB_PURE_CARD_UI__.latestReport.version,core.VERSION);
assert(v.nodes.message.textContent.includes('MC 不確定性超過門檻'));
assert.equal(v.nodes.btnPointAnalyze.disabled,false);
// No automatic next-hand labels without user confirmation.
v.nodes.btnPointAnalyze.click();assert.deepEqual(Array.from(v.jobs[1].input.settledRows),[]);
v.jobs[1].respond(good);
v.nodes.pureContinuous.checked=true;v.nodes.btnPointAnalyze.click();
assert.equal(v.jobs[2].input.settledRows.length,1);v.jobs[2].respond(good);
// Worker failure invalidates the current result, not fallback to old PF.
v.failWorker();v.nodes.btnPointAnalyze.click();assert.equal(v.nodes.directionText.textContent,'觀望');
assert.equal(v.root.__BBB_PURE_CARD_UI__.latestReport,null);
assert(v.nodes.message.textContent.includes('啟動失敗'));
// Illegal input clears old direction before any Worker can start.
const bad=mount();bad.nodes.directionText.textContent='莊';bad.nodes.pointPlayerFinal.value='5';
bad.nodes.pointBankerFinal.value='3+4';bad.nodes.btnPointAnalyze.click();
assert.equal(bad.nodes.directionText.textContent,'觀望');assert.equal(bad.jobs.length,0);
// Stale responses after explicit reset cannot restore a direction.
const stale=mount();stale.nodes.pointPlayerFinal.value='4+4';stale.nodes.pointBankerFinal.value='3+4';stale.nodes.btnPointAnalyze.click();
stale.nodes.btnBack.click();stale.jobs[0].respond(good);assert.equal(stale.root.__BBB_PURE_CARD_UI__.latestReport,null);
console.log('PASS panel: real report, invalid input, worker failure, no PF fallback, causal opt-in settlement, stale response rejection');

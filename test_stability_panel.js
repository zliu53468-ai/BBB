'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const base=require('./pf_eor_v4_runtime.js'),stable=require('./stability_runtime.js');
function mount(){
 const nodes={};for(const id of ['btnPointAnalyze','pointPlayerFinal','pointBankerFinal','directionText','directionCode','message','modePill','regime','strength','confidence','directionOrb','localCalibrationStatus','pointInputStatus','btnEnd','btnBack'])nodes[id]={value:'',textContent:'',disabled:false,handlers:{},addEventListener(k,f){this.handlers[k]=f;}};
 const jobs=[],data=new Map();class Worker{constructor(){jobs.push(this);}postMessage(v){this.input=v;}terminate(){this.terminated=true;}respond(report){this.onmessage({data:{id:this.input.id,report}});}}
 const document={getElementById:id=>nodes[id]||null,querySelectorAll:()=>[],querySelector:()=>null};
 const root={document,BBBPhysicalV4:base,BBBStability:stable,Worker,localStorage:{getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v)},matchMedia:()=>({matches:false})};
 vm.runInNewContext(fs.readFileSync('pf_eor_v4_panel.js','utf8'),{window:root,setTimeout,clearTimeout,console});
 return {root,nodes,jobs};
}
(async()=>{
 const view=mount(),ui=view.root.__BBB_PHYSICAL_V4_PANEL__;
 const pending=ui.submitPoints(5,3);assert.equal(view.nodes.directionText.textContent,'觀望');assert.equal(ui.latestReport,null);
 const report=new stable.StablePhysicalPointV4().analyze(5,3,{seed:7});view.jobs[0].respond(report);await pending;
 assert.equal(ui.latestReport.randomization.seed,7);assert.equal(ui.audit[0].analysis_seed,7);
 const second=ui.submitPoints(1,2);view.jobs[1].onerror();await second;assert.equal(ui.latestReport,null);assert.equal(view.nodes.directionText.textContent,'觀望');
 const stale=ui.submitPoints(5,3);view.nodes.btnEnd.handlers.click();await stale;
 view.jobs[2].respond(report);assert.equal(ui.latestReport,null);assert.equal(ui.audit.length,0);
 console.log('PASS asynchronous panel pending-SKIP, successful Worker report, error fail-closed, cancellation and stale-response rejection');
})().catch(e=>{console.error(e);process.exitCode=1;});

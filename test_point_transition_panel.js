'use strict';
const assert=require('node:assert/strict'),vm=require('vm'),fs=require('fs');
const api=require('./point_transition_runtime.js');
function mount(data=new Map()){
 const ids=['directionText','directionCode','confidence','strength','regime','message','directionOrb','localCalibrationStatus','pointInputStatus','btnPointAnalyze','pointPlayerFinal','pointBankerFinal','btnBack','btnNewSequence','btnEnd','modePill'];
 const nodes={};for(const id of ids)nodes[id]={value:'',textContent:'',disabled:false,handlers:{},addEventListener(k,fn){this.handlers[k]=fn;}};
 let fail=false;
 const root={BBBPointTransitions:api,localStorage:{getItem:k=>data.get(k)||null,setItem:(k,v)=>{if(fail)throw Error('write_failed');data.set(k,v);}},matchMedia:()=>({matches:false})};
 const document={getElementById:id=>nodes[id]||null,querySelectorAll:()=>[],querySelector:()=>null};
 vm.runInNewContext(fs.readFileSync('point_transition_panel.js','utf8'),{window:root,document,console,setTimeout,clearTimeout});
 return {root,nodes,data,setFailure(){fail=true;},async enter(p,b){nodes.pointPlayerFinal.value=String(p);nodes.pointBankerFinal.value=String(b);await root.__BBB_POINT_TRANSITION_UI__.submit();}};
}
(async()=>{
 const v=mount();await v.enter(5,3);await v.enter(2,8);
 assert.equal(v.root.__BBB_POINT_TRANSITION_UI__.snapshot.totalRounds,2);
 assert.equal(v.root.__BBB_POINT_TRANSITION_UI__.snapshot.matrix[53][28],1);
 const reloaded=mount(v.data);assert.equal(reloaded.root.__BBB_POINT_TRANSITION_UI__.snapshot.totalRounds,2);
 assert(reloaded.nodes.pointInputStatus.textContent.includes('已恢復'));
 await reloaded.enter(9,1);assert.equal(reloaded.root.__BBB_POINT_TRANSITION_UI__.snapshot.totalRounds,3);
 await reloaded.enter('J+5',3);assert.equal(reloaded.root.__BBB_POINT_TRANSITION_UI__.snapshot.totalRounds,3);
 assert.equal(reloaded.nodes.directionText.textContent,'觀望');
 reloaded.setFailure();await reloaded.enter(1,2);
 assert.equal(reloaded.root.__BBB_POINT_TRANSITION_UI__.snapshot.totalRounds,3);
 assert.equal(reloaded.nodes.directionText.textContent,'觀望');
 const corrupt=mount(new Map([['bgs_point_transition_history_v1','broken JSON']]));
 await corrupt.enter(1,2);assert.equal(corrupt.root.__BBB_POINT_TRANSITION_UI__.snapshot,null);
 assert(corrupt.nodes.message.textContent.includes('歷史載入異常'));
 console.log('PASS total-only UI, cumulative updates, actual persistence across reload, invalid input, failed-write rollback and corrupt-storage fail-closed');
})().catch(e=>{console.error(e);process.exitCode=1;});

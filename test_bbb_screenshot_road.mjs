#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
// Expose isolated pure scanner internals to the fixture, not to the live page.
let source=fs.readFileSync("bbb-screenshot-road.js","utf8");
const anchor="root.__BBB_ROAD_SCANNER__={";
assert(source.includes(anchor));
source=source.replace(anchor,"root.__BBB_TEST_INTERNALS__={hslMask,reconstruct,scoreRegion,importProposal,setSource:c=>sourceCanvas=c,setProposal:s=>proposal=s};\n"+anchor);
const nodes=new Map(),states={history:[],start:0};
const node=id=>{
  if(!nodes.has(id))nodes.set(id,{value:"",textContent:"",disabled:false,hidden:false,
    classList:{toggle(){},add(){}},replaceChildren(){},appendChild(){},clientWidth:500,
    addEventListener(){},click(){}});
  return nodes.get(id);
};
const resultIds={btnB:"B",btnP:"P",btnT:"T"};
for(const [id,outcome] of Object.entries(resultIds))node(id).click=()=>states.history.push(outcome);
node("btnStart").click=()=>states.start++;
const localStorage={getItem(k){return k==="bgs256d_frozen_6x15_forward_v18"?JSON.stringify({history:states.history}):null;}};
const root={localStorage,__BGS_FINAL56__:{},__BGS256_TEST__:{},
  __BBB_LICENSE_STORAGE__:{status:()=>({pending_writes:0,error:""})}};
const document={readyState:"complete",getElementById:id=>["macroPlayerScore","macroBankerScore","bbbLastScore"].includes(id)?null:node(id),addEventListener(){},createElement(tag){
  return {type:tag,className:"",textContent:"",addEventListener(){},title:""};
}};
const context={window:root,document,performance:{now:()=>0},localStorage,
  Image:class{},URL:{},Math,Array,Set,Map,Promise,Error,RegExp,String,Number,
  console,setTimeout,clearTimeout};
vm.runInNewContext(source,context,{filename:"bbb-screenshot-road.js"});
const {hslMask,reconstruct,scoreRegion,importProposal,setSource,setProposal}=root.__BBB_TEST_INTERNALS__;
assert.equal(hslMask(230,25,28),1,"BGS red maps to banker");
assert.equal(hslMask(25,80,240),2,"BGS blue maps to player");
assert.equal(hslMask(12,200,40),4,"BGS green maps to tie marker");
assert.equal(hslMask(236,236,236),0,"background not interpreted as a result");
const outcomes=["B","B","P","P","B","B","P","B","P","P","B","P","B"];
const rings=[];
let col=-1,row=0,previous="";
for(const outcome of outcomes){
  if(previous===outcome)row++;else{col++;row=0;}
  rings.push({col,row,outcome});previous=outcome;
}
const width=15*18,height=6*18;
const pixels=new Uint8ClampedArray(width*height*4);
for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  let rgb=[235,235,235];
  for(const c of rings){
    const dx=x-(c.col*18+9),dy=y-(c.row*18+9);
    if(dx*dx+dy*dy<36)rgb=c.outcome==="B"?[216,34,47]:[36,93,222];
  }
  const pos=(y*width+x)*4;pixels.set([...rgb,255],pos);
}
const canvas={width,height,getContext:()=>({getImageData:(x,y,w,h)=>{
  const arr=new Uint8ClampedArray(w*h*4);
  for(let j=0;j<h;j++)for(let i=0;i<w;i++)
    arr.set(pixels.subarray(((y+j)*width+x+i)*4,((y+j)*width+x+i+1)*4),(j*w+i)*4);
  return {data:arr};
}})};
setSource(canvas);
const scan=scoreRegion([0,0,1,1],15);
assert(scan,"synthetic 6x15 BGS big-road fixture recognized");
assert.equal(scan.outcomes.join(""),outcomes.join(""),"chronological reconstruction matches each original banker/player result");
assert.equal(scan.recognized,outcomes.length,"no phantom dots from grid background");
setProposal(outcomes.slice());
await importProposal();
assert.equal(states.history.join(""),outcomes.join(""),"confirmed scan goes through authoritative BBB result buttons");
assert.equal(states.start,1,"confirmed scan automatically analyzes next hand");
assert.equal(root.__BBB_ROAD_SCANNER__.source.includes("BGS"),true);
const size=states.history.length;
setProposal(["B","P","T"]);
await importProposal();
assert.equal(states.history.length,size,"scan cannot replace or append to an active shoe");
assert.match(node("bbbScanMessage").textContent,/不允許覆蓋或混合/);
console.log(JSON.stringify({ok:true,checks:["BGS red blue green masks","6x15 chronology reconstruction","manual scan confirmation","reject merging a second screenshot"]}));

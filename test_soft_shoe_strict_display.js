"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const s=require("./soft_shoe_decision_runtime.js");
const cfg=s.config(),off=s.config({STRICT_DUAL_DISPLAY_ENABLED:false});
const view=side=>({eligible:true,candidate:side,direction:side});
const road=side=>({available:true,road_side:side,adaptive:{w_applied:.03,stability:.45}});
const path={future_edge_score:.60};
const display=(v,r,p,c,options=cfg)=>s.strictDualDisplay(v,r,p,c,options);
for(const [key,value] of Object.entries({STRICT_DUAL_DISPLAY_ENABLED:true,
 STRICT_DUAL_MIN_W:.03,STRICT_DUAL_MIN_STABILITY:.45,STRICT_DUAL_MIN_EDGE:.60}))
 assert.equal(s.DEFAULT[key],value);
assert.throws(()=>s.config({STRICT_DUAL_DISPLAY_ENABLED:"false"}),/invalid_STRICT_DUAL_DISPLAY_ENABLED/);
for(const key of ["STRICT_DUAL_MIN_W","STRICT_DUAL_MIN_STABILITY","STRICT_DUAL_MIN_EDGE"])
 for(const value of [-.01,1.01,NaN,Infinity,"0.6",null])
  assert.throws(()=>s.config({[key]:value}),new RegExp("invalid_"+key));

// 1. Off restores the original view, even if every road requirement fails.
for(const direction of ["B","P","SKIP"]){
 const result=display({eligible:direction!=="SKIP",direction},null,null,"B",off);
 assert.equal(result.direction,direction);
 assert.equal(result.enabled,false);
 assert.deepEqual(result.fail_reasons,[]);
}

// 2. Soft eligible alone is insufficient. Missing/invalid metrics fail closed.
const failures=[
 [{...view("B"),eligible:false},road("B"),path,"B","soft_not_eligible"],
 [view("B"),road("B"),path,"T","invalid_candidate"],
 [view("B"),{...road("B"),available:false},path,"B","road_unavailable"],
 [view("B"),road("P"),path,"B","road_not_aligned"],
 [view("B"),{...road("B"),adaptive:{w_applied:.029999,stability:.45}},path,"B","road_weight_below_min"],
 [view("B"),{...road("B"),adaptive:{w_applied:.03,stability:.449999}},path,"B","road_stability_below_min"],
 [view("B"),road("B"),{future_edge_score:.599999},"B","future_edge_below_min"],
 [view("B"),{...road("B"),adaptive:{}},path,"B","road_weight_below_min"],
 [view("B"),{...road("B"),adaptive:{w_applied:NaN,stability:Infinity}},path,"B","road_stability_below_min"],
 ...[null,undefined,NaN,Infinity,"0.60"].map(score=>
  [view("B"),road("B"),{future_edge_score:score},"B","future_edge_below_min"])
];
for(const [v,r,p,c,reason] of failures){
 const before=JSON.stringify({v,r,p,c}),result=display(v,r,p,c);
 assert.equal(result.direction,"SKIP");assert.equal(result.label,"觀望");
 assert.equal(result.passed,false);assert(result.fail_reasons.includes(reason),reason);
 assert.equal(JSON.stringify({v,r,p,c}),before,"display must not mutate inputs");
}

// 3. Inclusive boundaries allow B/P; an absent optional future score is allowed.
for(const side of ["B","P"])
 for(const future of [path,{}, {available:false}])
  assert.deepEqual(display(view(side),road(side),future,side),{
   enabled:true,direction:side,label:"雙訊號同向",passed:true,fail_reasons:[]});

// Full run integration: only strict fields/config are added by this switch.
const histories=[[],["B"],list("BPPBTBBPPT"),list("BP".repeat(12))];
function list(text){return [...text];}
function withoutDisplay(report){
 const copy=JSON.parse(JSON.stringify(report));
 for(const key of ["strict_display","strict_display_direction","strict_display_label"])
  delete copy[key];
 for(const key of Object.keys(copy.config))if(key.startsWith("STRICT_DUAL_"))delete copy.config[key];
 return copy;
}
for(const history of histories){
 const on=s.run(history),legacy=s.run(history,{STRICT_DUAL_DISPLAY_ENABLED:false});
 assert.deepEqual(withoutDisplay(on),withoutDisplay(legacy));
 assert.equal(legacy.strict_display_direction,legacy.research_view.direction);
 assert.equal(on.action,"Skip");assert.equal(on.direction,"SKIP");
 assert.equal(on.bankroll_fraction,0);assert.equal(on.validated_edge,false);
 assert.equal(on.wager_gate.eligible,false);
 assert.deepEqual(on.strict_display,display(on.research_view,on.road_aux,on.future_path,on.candidate));
 assert.equal(on.strict_display_direction,on.strict_display.direction);
 assert.equal(on.strict_display_label,on.strict_display.label);
}
const unavailable=s.run(Array(73).fill("B"));
assert.equal(unavailable.action,"Skip");
assert.equal(unavailable.strict_display_direction,"SKIP");
assert.equal(unavailable.strict_display.passed,false);

// Execute the actual panel with an eligible old view and a strict SKIP.
async function panelDirection(report){
 const elements=new Map();
 const get=id=>{
  if(!elements.has(id))elements.set(id,{textContent:"",className:"",hidden:false,
   addEventListener(){}});
  return elements.get(id);
 };
 class Worker{
  terminate(){}
  postMessage({id}){this.onmessage({data:{id,report}});}
 }
 const root={BBBSoftShoe:s,BBBBigRoad:{render(){}},Worker,
  document:{getElementById:get,querySelector:()=>null},
  localStorage:{getItem:()=>null,setItem(){}}};
 vm.runInNewContext(fs.readFileSync("soft_shoe_panel.js","utf8"),
  {window:root,setTimeout,clearTimeout});
 await root.__BBB_SOFT_SHOE_PANEL__.recalculate();
 return {direction:get("directionText").textContent,label:get("regime").textContent};
}
async function checkPanel(){
 const base={version:s.VERSION,candidate:"B",research_view:view("B"),
  config:cfg,strict_display_direction:"SKIP",
  strict_display:{enabled:true,direction:"SKIP",label:"觀望",passed:false}};
 assert.deepEqual(await panelDirection(base),{direction:"觀望",label:"觀望"});
 assert.equal((await panelDirection({...base,config:off})).direction,"莊");
 for(const side of ["B","P"]){
  const strict=display(view(side),road(side),path,side);
  assert.deepEqual(await panelDirection({...base,candidate:side,research_view:view(side),
   strict_display:strict,strict_display_direction:side}),
   {direction:side==="B"?"莊":"閒",label:"雙訊號同向"});
 }
 console.log("Strict dual display: rollback, rejection, B/P boundaries, report and panel passed");
}
checkPanel().catch(error=>{console.error(error);process.exitCode=1;});

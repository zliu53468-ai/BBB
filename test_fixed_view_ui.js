"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const html=fs.readFileSync("index.html","utf8");
const css=fs.readFileSync("styles.css","utf8");
const script=fs.readFileSync("viewport-fit.js","utf8");

// Do not remove bankroll controls or any prediction input; remove side bets only.
for(const text of ["side-bet-card","sideBetTitle","sideBetNote","本局邊注建議","幸運六"])
  assert(!html.includes(text),"obsolete side-bet widget remains: "+text);
for(const id of ["predictionApp","appViewportContent","bankrollSelect","stakePercent",
  "btnStart","btnB","btnP","btnT","btnBack","btnEnd","licenseGate","licenseForm",
  "btnExportTraining","btnExportObservations"])
  assert(html.includes('id="'+id+'"'),"core page control missing: "+id);
assert(html.indexOf("brand-hero")<html.indexOf("appViewportContent"),
  "fixed hero must stay outside scaled content");
assert(html.includes("auth-client.js?v=20261101-v2-pf96-l256-m96"),
  "prediction loader must not be altered");
assert(html.includes("viewport-fit.js?v=20261102-fixed-view-no-sidebets-v1"));
assert(html.includes("styles.css?v=20261102-fixed-view-no-sidebets-v1"));
assert(css.includes("html.bbb-viewport-locked")&&css.includes("body.bbb-viewport-locked"));
assert(css.includes("overflow:hidden !important"));
assert(css.includes("height:100dvh"));
assert(css.includes("transform:scale(var(--bbb-fit-scale,1))"));

// Run the layout controller in isolation. It may only change CSS state.
const rootClasses=new Set(),bodyClasses=new Set(),observers=[];
const classList=target=>({toggle(name,value){
  if(value)target.add(name);else target.delete(name);
}});
const app={hidden:true},content={
  scrollHeight:1120,offsetHeight:1120,
  style:{values:{},setProperty(k,v){this.values[k]=Number(v);},
    removeProperty(k){delete this.values[k];}}
};
const document={
  getElementById(k){return k==="predictionApp"?app:k==="appViewportContent"?content:null;},
  documentElement:{clientHeight:700,classList:classList(rootClasses)},
  body:{classList:classList(bodyClasses)},
  addEventListener(){}
};
const window={
  innerHeight:700,visualViewport:{height:700,addEventListener(){}},
  addEventListener(){},requestAnimationFrame(callback){callback();}
};
class Observer{
  constructor(callback){this.callback=callback;observers.push(this);}
  observe(){}
}
vm.runInNewContext(script,{document,window,MutationObserver:Observer,ResizeObserver:Observer});
assert(!rootClasses.has("bbb-viewport-locked"),"login must stay scrollable");
app.hidden=false;observers[0].callback();
assert(rootClasses.has("bbb-viewport-locked")&&bodyClasses.has("bbb-viewport-locked"));
assert(Math.abs(content.style.values["--bbb-fit-scale"]-(688/1120))<1e-8,
  "tall content must fit by scale rather than clipping");
content.scrollHeight=550;content.offsetHeight=550;observers[1].callback();
assert.equal(content.style.values["--bbb-fit-scale"],1,
  "already fitting screen must preserve full size");
app.hidden=true;observers[0].callback();
assert(!rootClasses.has("bbb-viewport-locked")&&!bodyClasses.has("bbb-viewport-locked"),
  "exit/login must unlock document scrolling");
console.log("PASS no-scroll single viewport, scaled background independence, side-bet removal, original controls and prediction loader");

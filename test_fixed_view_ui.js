"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const html=fs.readFileSync("index.html","utf8");
const css=fs.readFileSync("styles.css","utf8");
const script=fs.readFileSync("viewport-fit.js","utf8");
for(const s of ["side-bet-card","sideBetTitle","sideBetNote","本局邊注建議","幸運六"])
  assert(!html.includes(s),"removed side-bet text must stay removed: "+s);
for(const id of ["predictionApp","appViewportContent","bankrollSelect","stakePercent",
  "btnStart","btnB","btnP","btnT","btnBack","btnEnd","licenseGate","licenseForm",
  "btnExportTraining","btnExportObservations"])
  assert(html.includes('id="'+id+'"'),"missing original control: "+id);
assert(html.indexOf("brand-hero")<html.indexOf("appViewportContent"),
  "background must remain outside scroll content");
assert(html.includes("auth-client.js?v=20261103-dynamic-shoe-guard-v1"),
  "production prediction core loader must not be changed");
assert(html.includes("viewport-fit.js?v=20261102-mobile-stable-v2"),"stable viewport loader version");
assert(html.includes("styles.css?v=20261102-mobile-stable-v2"),"stable CSS cache key");

// A frozen outer viewport with an accessible independently scrolling content
// region replaces the old unbounded scaled page. Especially important for
// iPhone app browsers and larger Dynamic Type font settings.
assert(css.includes("height:100svh"),"use stable small viewport, not dynamic toolbars");
assert(css.includes("overflow-y:auto"),"short screens must reach all controls");
assert(css.includes("overflow:hidden!important"),"outer document remains fixed");
assert(css.includes("-webkit-overflow-scrolling:touch"),"iOS internal scrolling");
assert(css.includes("position:sticky"),"header stays within scroll frame");
assert(css.includes("grid-template-columns:minmax(0,1fr) auto"),
  "mobile status must receive its own row");
assert(css.includes("#appViewportContent #modePill"),"status truncation applies");
assert(css.includes("text-overflow:ellipsis"),"long mode label must not stack");
assert(!css.includes("--bbb-fit-scale"),"remove presentation scale feedback loop");
assert(!css.includes("transform:scale(var("),"whole panel cannot shrink");
assert(!script.includes("visualViewport")&&!script.includes("ResizeObserver")&&
  !script.includes("requestAnimationFrame"),"layout must not resize repeatedly");
assert(!script.includes("scrollHeight")&&!script.includes("offsetHeight"),
  "do not compute changing content height");

// Verify the only JavaScript side effect is toggling scroll lock on auth.
const htmlClasses=new Set(),bodyClasses=new Set(),watchers=[];
const getClasses=set=>({toggle(name,active){
  active?set.add(name):set.delete(name);
}});
const app={hidden:true};
const dom={
  getElementById:id=>id==="predictionApp"?app:null,
  documentElement:{classList:getClasses(htmlClasses)},
  body:{classList:getClasses(bodyClasses)}
};
class Observer{
  constructor(callback){this.callback=callback;watchers.push(this);}
  observe(target,opt){
    assert.strictEqual(target,app);
    assert.deepEqual(Array.from(opt.attributeFilter),["hidden"]);
  }
}
vm.runInNewContext(script,{document:dom,MutationObserver:Observer});
assert(!htmlClasses.has("bbb-viewport-locked")&&!bodyClasses.has("bbb-viewport-locked"),
  "login needs normal scroll when keyboard is open");
assert.equal(watchers.length,1,"watch only authorization visibility");
app.hidden=false;watchers[0].callback();
assert(htmlClasses.has("bbb-viewport-locked")&&bodyClasses.has("bbb-viewport-locked"),
  "an unlocked app must lock the outer document");
app.hidden=true;watchers[0].callback();
assert(!htmlClasses.has("bbb-viewport-locked")&&!bodyClasses.has("bbb-viewport-locked"),
  "logout must restore login scroll");
console.log("PASS mobile stable viewport: no automatic scaling, usable small-screen scrolling, nonstacked status, original inputs, unchanged prediction loader");

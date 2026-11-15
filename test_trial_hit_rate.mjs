// Truthful completed-prediction 60% trial cutoff regression.
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const code=read("trial-hit-rate.js");
const panel=read("soft_shoe_panel.js");
const html=read("index.html");
const end=read("trial-ended.html");
const money=read("bankroll-trial.js");
const trial=read("trial-client.js");
assert(html.includes('id="bbbTrialAccuracyPanel"'),"metric shown in trial");
assert(html.includes("至少 10 次有效預測結算且命中率達 60%"),"upfront cutoff condition clear");
assert(html.includes('src="trial-hit-rate.js?v=20261115-accuracy-trigger-v1"'),"tracker loaded before trial");
assert(html.indexOf('src="trial-hit-rate.js')<html.indexOf('src="trial-client.js'),"tracker installed before dynamic panel");
assert(trial.includes("window.BBB_TRIAL_ACCURACY?.onTrialMounted"),"tracker initialized after the panel");
assert(panel.includes('view.eligible===true'),"panel only scores quality-gated direction");
assert(panel.includes("last?.observations===before.length"),"prediction must correspond to previous actual recorded history");
assert(panel.includes('notifyTrial("bbb:trial-back"'),"undo event");
assert(panel.includes('notifyTrial("bbb:trial-reset"'),"reset event");
assert(!money.includes('reason=profit60'),"capital target must not end hit-rate trial");
assert(end.includes('reason==="accuracy60"'),"ended page supports explicit accuracy reason");
assert(end.includes('data-bbb-link="casinoRegisterUrl"')&&end.includes('data-bbb-link="lineContactUrl"'),"existing exit links remain");

function simulated(storage=new Map(),observed=[]){
  const events={};
  const pending=new Map();
  let timerId=0;
  const redirects=[];
  const elById=Object.fromEntries([
    "bbbTrialAccuracyPanel","bbbTrialAccuracyHits","bbbTrialAccuracyValid",
    "bbbTrialAccuracyRate","bbbTrialAccuracyStatus"
  ].map(id=>[id,{textContent:"",hidden:true}]));
  const document={getElementById(id){return elById[id]||null;}};
  const window={
    __BBB_SOFT_SHOE_PANEL__:{history:observed.slice()},
    location:{replace(path){redirects.push(path);}},
    addEventListener(name,cb){events[name]=cb;}
  };
  const sessionStorage={
    getItem(key){return storage.get(key)||null;},
    setItem(key,val){storage.set(key,val);}
  };
  const timeout=(cb,ms)=>{const id=++timerId;pending.set(id,{cb,ms});return id;};
  const cancel=id=>pending.delete(id);
  vm.runInNewContext(code,{window,document,sessionStorage,setTimeout:timeout,clearTimeout:cancel,Math,JSON,Number,Array},{filename:"trial-hit-rate.js"});
  return {
    controller:window.BBB_TRIAL_ACCURACY,
    emit(name,detail){assert(events[name],"missing "+name);events[name]({detail});},
    els:elById,redirects,storage,
    runPending(){for(const [id,t]of [...pending]){pending.delete(id);t.cb();}},
    scheduled(){return [...pending.values()];}
  };
}
const m=simulated();m.controller.onTrialMounted({new_session:true});
assert.equal(m.els.bbbTrialAccuracyPanel.hidden,false);
assert.equal(m.controller.snapshot().valid,0);
assert.equal(m.els.bbbTrialAccuracyRate.textContent,"—");
let history=[];
function record(outcome,prediction){
  const before=history.slice();history.push(outcome);
  m.emit("bbb:trial-result",{before,after:history.slice(),outcome,prediction});
}
record("T","B");
record("P",null);
assert.equal(m.controller.snapshot().valid,0,"ties and abstentions excluded");
assert.equal(m.els.bbbTrialAccuracyValid.textContent,"0");
for(let i=0;i<9;i++){
  record(i<6?"B":"P","B");
  assert.equal(m.scheduled().length,0,"less than 10 resolved predictions never ends trial");
}
assert.equal(m.controller.snapshot().valid,9);
assert.equal(m.controller.snapshot().hits,6);
record("P","B");
assert.equal(m.controller.snapshot().valid,10);
assert.equal(m.controller.snapshot().hits,6);
assert.equal(m.els.bbbTrialAccuracyRate.textContent,"60.0%");
assert.equal(m.scheduled().length,1,"10th settled direction with 60% hit rate schedules redirect");
m.emit("bbb:trial-back",{after:history.slice(0,-1)});
history.pop();
assert.equal(m.controller.snapshot().valid,9,"undo removes previous scored outcome");
assert.equal(m.scheduled().length,0,"undo cancels premature redirect");
record("P","B");
assert.equal(m.scheduled().length,1);
const snapshotBeforeReload=history.slice();
const r=simulated(m.storage,snapshotBeforeReload);
r.controller.onTrialMounted({new_session:false});
assert.equal(r.controller.snapshot().valid,10,"refresh restores accurately matched ledger");
assert.equal(r.controller.snapshot().hits,6);
assert.equal(r.scheduled().length,1);
r.runPending();
assert.deepEqual(r.redirects,["./trial-ended.html?reason=accuracy60"],"real 60% cutoff opens existing exit screen");
const different=simulated(m.storage,[]);
different.controller.onTrialMounted({new_session:false});
assert.equal(different.controller.snapshot().valid,0,"mismatched history cannot claim historical wins");
const fresh=simulated(m.storage,[]);
fresh.controller.onTrialMounted({new_session:true});
assert.equal(fresh.controller.snapshot().valid,0,"new trial clears old ledger");
console.log(JSON.stringify({ok:true,checks:["eligible_previous_round","skip_and_tie_excluded","minimum_ten","60_percent_exact","undo_cancels_redirect","refresh_restores_matching_ledger","mismatched_history_not_scored","new_trial_resets","keeps_DG_and_casino_links"]}));

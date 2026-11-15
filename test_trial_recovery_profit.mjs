// Trial recovery and 60%-profit redirect regression test.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const trialCode=read("trial-client.js");
const moneyCode=read("bankroll-trial.js");
const html=read("index.html");
const ended=read("trial-ended.html");
const worker=read("license-worker/src/index.js");
function element(id){
  const handlers={};
  return {
    id,hidden:false,open:false,disabled:false,textContent:"",value:"",
    style:{},classList:{toggle(){}},
    addEventListener(name,cb){handlers[name]=cb;},
    trigger(name){handlers[name]?.({preventDefault(){}});},
    click(){handlers.click?.({preventDefault(){}});},
    setAttribute(){},removeAttribute(){},hasAttribute(){return false;},
    focus(){},showModal(){this.open=true;},close(){this.open=false;},
    get handlers(){return handlers;}
  };
}
const ids=["trialStartBtn","trialStartMessage","licenseGate","predictionApp",
 "bbbTrialBanner","bbbTrialCountdown","bbbTrialStatus","licenseMessage","btnEnd",
 "bbbBankrollDialog","bbbBankrollForm","bbbInitialBankroll","bbbBankrollTargetPreview",
 "bbbBankrollError","bbbBankrollCancel","bbbTrialBankrollPanel",
 "bbbMoneyUpdateForm","bbbMoneyUpdateAmount","bbbMoneyStart","bbbMoneyTarget",
 "bbbMoneyCurrent","bbbMoneyKelly","bbbMoneyStake","bbbMoneyStatus",
 "bbbMoneyProgress"];
const dom=Object.fromEntries(ids.map(id=>[id,element(id)]));
dom.licenseGate.hidden=false;
dom.predictionApp.hidden=true;
dom.bbbTrialBanner.hidden=true;
dom.licenseMessage.textContent="首次使用請輸入開通碼";
const keys=new Map(),session=new Map(),redirects=[],called=[],timers=[];
const storage=map=>({getItem:key=>map.get(key)||null,setItem:(key,value)=>map.set(key,value),removeItem:key=>map.delete(key)});
const document={
 getElementById(id){return dom[id]||null;},
 querySelector(selector){return selector===".bbb-profit-progress"?element("progress"):null;},
 createElement(){return {async:false,onload:null,onerror:null,src:""};},
 body:{appendChild(el){queueMicrotask(()=>el.onload?.());}},
 addEventListener(){}
};
const location={href:"https://bgsv1.xyz/",replace(url){redirects.push(url);}};
const window={
 BBB_AUTH_API_BASE:"https://bbb-license-gateway.zliu53468.workers.dev",
 crypto:{randomUUID(){return "550e8400-e29b-41d4-a716-446655440000";}},
 location,history:{state:null,replaceState(){}},
 addEventListener(){},open(){throw Error("BBB start must NOT open DG or external sites");}
};
const ctx={
 window,document,location,crypto:window.crypto,
 localStorage:storage(keys),sessionStorage:storage(session),
 URL,URLSearchParams,Date,Math,Number,JSON,performance:{now(){return 1000;}},
 fetch:async(url,request)=>{
   called.push({url,method:request.method});
   assert(url.endsWith("/api/trial/start"),"only trial start requested");
   return {ok:false,status:404,json:async()=>({error:"not_found"})};
 },
 setInterval(){return 7;},clearInterval(){},
 setTimeout(fn,ms){timers.push({fn,ms});return timers.length;},
 clearTimeout(){},queueMicrotask
};
vm.runInNewContext(moneyCode,ctx,{filename:"bankroll-trial.js"});
vm.runInNewContext(trialCode,ctx,{filename:"trial-client.js"});
dom.trialStartBtn.click();
assert.equal(dom.bbbBankrollDialog.open,true,"first trial click only shows bankroll form");
assert.equal(called.length,0,"API not called before bankroll is confirmed");
dom.bbbInitialBankroll.value="10000";
dom.bbbBankrollForm.trigger("submit");
await new Promise(resolve=>setTimeout(resolve,45));
assert.equal(called.length,1,"trial API 404 was detected");
const local=JSON.parse(keys.get("bbb-public-trial-local-fallback-v1"));
assert.equal(local.expires_at-local.started_at,30*60*1000,"local fallback fixed to 30 mins");
assert.equal(dom.predictionApp.hidden,false,"fallback opens BBB panel");
assert.equal(dom.licenseGate.hidden,true,"landing hidden only after explicit user action");
assert.equal(dom.bbbTrialBanner.hidden,false,"trial countdown visible");
assert(dom.bbbTrialStatus.textContent.includes("本瀏覽器"),"local fallback disclosed");
assert.equal(redirects.length,0,"not_found should not send user to end or DG");
assert.equal(window.BBB_BANKROLL.getSnapshot().target,16000,"+60% profit target");
dom.bbbMoneyUpdateAmount.value="15999";
dom.bbbMoneyUpdateForm.trigger("submit");
assert.equal(timers.filter(x=>x.ms===1800).length,0,"below target no redirect");
dom.bbbMoneyUpdateAmount.value="16000";
dom.bbbMoneyUpdateForm.trigger("submit");
assert.equal(window.BBB_BANKROLL.getSnapshot().hit,true,"target hit");
const redirectTimer=timers.find(x=>x.ms===1800);
assert(redirectTimer,"successful manual update queues redirect");
redirectTimer.fn();
assert.equal(redirects[0],"./trial-ended.html?reason=profit60","reuse trial ended conversion page");
assert(ended.includes('reason")=="profit60"')||ended.includes('get("reason")==="profit60"'),"end page explains reason");
assert(ended.includes('data-bbb-link="casinoRegisterUrl"')&&ended.includes('data-bbb-link="lineContactUrl"'),"end page has both casino and LINE");
assert(!trialCode.includes("window.open("),"BBB trial never auto-opens DG");
assert(html.includes('data-bbb-link="dgTrialUrl"'),"DG still independent button");
assert(worker.includes("trial_api_version:1"),"deployed Worker is version-identifiable");
console.log(JSON.stringify({ok:true,checks:["bankroll_before_trial","404_not_found_local_fallback","no_DG_popup","fixed_30min","profit_60_manual_balance","no_redirect_below_target","redirect_at_target","casino_and_LINE_end_page","worker_version_health"]}));

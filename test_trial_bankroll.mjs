// BBB 60%-of-initial-bankroll stop-win, model-gated fractional Kelly, DG separation, SEO.
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
const read=name=>fs.readFileSync(new URL("./"+name,import.meta.url),"utf8");
const html=read("index.html"),bank=read("bankroll-trial.js"),trial=read("trial-client.js");
const guide=read("baccarat-road-guide.html"),map=read("sitemap.xml");
assert(/<title>[^<]*莊家剋星[^<]*百家樂[^<]*<\/title>/.test(html),"Search title reflects official 莊家剋星 brand and baccarat content");
assert(html.includes("百家樂大路")&&html.includes("百家樂路單"),"natural on-page keywords");
assert(guide.includes('rel="canonical" href="https://bgsv1.xyz/baccarat-road-guide.html"'),"guide canonical");
assert(map.includes("https://bgsv1.xyz/baccarat-road-guide.html"),"guide in sitemap");
assert(html.includes('id="bbbBankrollDialog"')&&html.includes('id="bbbInitialBankroll"'),"capital input exists BEFORE panel");
assert(html.indexOf('id="bbbBankrollDialog"')<html.indexOf('id="predictionApp"'),"capital form is before app");
assert(html.includes('id="bbbTrialBankrollPanel"'),"stop-win dashboard exists");
assert(!html.includes('id="dgDirectTrialLink"')&&html.includes('id="trialStartBtn"'),"user-requested DG removal preserves BBB trial");
assert(!trial.includes("openDgOnUserClick"),"BBB start does not open DG");
assert(trial.includes("bankroll.promptAndStart()"),"trial requires capital setup");
assert(trial.includes("bankroll.consumeConfirmation()"),"trial only starts after capital confirmation");
assert(trial.includes("window.BBB_BANKROLL?.onTrialMounted()"),"dashboard starts after approved trial mounting");
assert(bank.includes("report.bankroll_fraction"),"Kelly fraction comes from original engine");
assert(bank.includes("report?.validated_edge===true"),"no non-gated advice");
assert(bank.includes('updateText("bbbMoneyStake",money(0))'),"zero advice after stop win");
assert(!html.includes('id="bbbStopLoss"')&&!bank.includes("stop_loss"),"no stop-loss field introduced");

const ids=[
 "bbbBankrollDialog","bbbBankrollForm","bbbInitialBankroll","bbbBankrollTargetPreview",
 "bbbBankrollError","bbbBankrollCancel","trialStartBtn","bbbTrialBankrollPanel",
 "bbbMoneyUpdateForm","bbbMoneyUpdateAmount","bbbMoneyStart","bbbMoneyTarget",
 "bbbMoneyCurrent","bbbMoneyKelly","bbbMoneyStake","bbbMoneyProgress","bbbMoneyStatus"
];
const elements=Object.fromEntries(ids.map(id=>[id,{
  id,value:"",textContent:"",hidden:true,style:{},listeners:{},
  classList:{set:new Set(),toggle(n,on){if(on)this.set.add(n);else this.set.delete(n);}},
  addEventListener(type,fn){this.listeners[type]=fn;},
  setAttribute(name,value){this[name]=value;},
  focus(){},
  showModal(){this.open=true;},close(){this.open=false;},
  click(){this.clicked=(this.clicked||0)+1;},
  submit(){this.listeners.submit?.({preventDefault(){}});}
}]));
const progress={setAttribute(name,value){this[name]=value;}};
const store=new Map();
const sessionStorage={getItem(k){return store.get(k)||null;},setItem(k,v){store.set(k,v);}};
const document={getElementById(id){return elements[id]||null;},querySelector(selector){return selector===".bbb-profit-progress"?progress:null;}};
const timers={};
const redirects=[];
const window={__BBB_SOFT_SHOE_PANEL__:{latestReport:null},location:{replace(path){redirects.push(path);}}};
vm.runInNewContext(bank,{document,window,sessionStorage,setInterval(fn){timers.run=fn;return 1;},clearInterval(){},setTimeout(fn,ms){timers.redirect=fn;timers.delay=ms;return 2;},clearTimeout(){},console,Number,Math,Date,JSON,Intl});
const controller=window.BBB_BANKROLL;
assert(controller,"bankroll controller installed");
assert.equal(controller.consumeConfirmation(),false,"cannot enter without capital");
controller.promptAndStart();
assert.equal(elements.bbbBankrollDialog.open,true);
elements.bbbInitialBankroll.value="10000";
elements.bbbBankrollForm.submit();
assert.equal(elements.trialStartBtn.clicked,1,"after confirming, triggers BBB trial button");
assert.equal(controller.consumeConfirmation(),true,"one-time bankroll confirmation accepted");
assert.equal(controller.consumeConfirmation(),false,"cannot reuse confirmation");
controller.onTrialMounted();
assert.equal(controller.getSnapshot().target,16000,"60% goal = 1.6 times initial capital");
assert.equal(elements.bbbMoneyStart.textContent,"NT$ 10,000");
assert.equal(elements.bbbMoneyTarget.textContent,"NT$ 16,000");
assert.equal(elements.bbbMoneyKelly.textContent,"0% · 觀望","no qualified EV means no Kelly bet");
window.__BBB_SOFT_SHOE_PANEL__.latestReport={direction:"B",action:"Bet Banker",validated_edge:true,bankroll_fraction:.04};
timers.run();
assert.equal(elements.bbbMoneyKelly.textContent,"4.00%（模型分數凱莉）");
assert.equal(elements.bbbMoneyStake.textContent,"NT$ 400","4% of current capital");
elements.bbbMoneyUpdateAmount.value="16000";
elements.bbbMoneyUpdateForm.submit();
assert.equal(controller.getSnapshot().hit,true,"profit target marks session stop-win");
assert.equal(elements.bbbMoneyStake.textContent,"NT$ 0","no further Kelly amount after stop-win");
assert.equal(progress["aria-valuenow"],"100");
elements.bbbMoneyUpdateAmount.value="7000";
elements.bbbMoneyUpdateForm.submit();
assert.equal(controller.getSnapshot().hit,true,"stop-win latches until new bankroll setup");
assert.equal(elements.bbbMoneyStake.textContent,"NT$ 0","post-target stake remains zero");
assert.equal(timers.delay,undefined,"bankroll target does not auto-end prediction trial");
assert.equal(redirects.length,0,"60% bankroll profit is not the 60% prediction hit-rate condition");
console.log(JSON.stringify({ok:true,checks:["separate_dg","capital_required","60_percent_target","kelly_model_gate","zero_on_skip","stop_win_latched","no_auto_end_on_bankroll_profit","seo_title_and_guide"]}));

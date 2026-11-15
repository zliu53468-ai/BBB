// Isolated regression tests for five registration URLs and LINE transfer prompt.
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
const siteJs=fs.readFileSync(new URL("./site-links.js",import.meta.url),"utf8");
const binderJs=fs.readFileSync(new URL("./outbound-links.js",import.meta.url),"utf8");
const html=fs.readFileSync(new URL("./trial-ended.html",import.meta.url),"utf8");
const expected=[
  "https://kc0075.3a1788.bet/",
  "https://tu123.app/a/register?promoCode=kc0075",
  "https://kc0075.welove777.com/",
  "https://s.leyo.tw/r?p=6uyzkpuUobqV",
  "https://cc1688002.gm1688.net/"
];
assert(html.includes('id="casinoAccountDialog"'));
assert(html.includes("已有這間娛樂城帳號嗎？"));
assert(html.includes("聯繫官方管理員轉線"));
function makeElement(key){
  const attributes={"data-bbb-link":key};
  const classes=new Set(),listeners={};
  return {
    getAttribute(name){return attributes[name]??null;},
    setAttribute(name,value){attributes[name]=value;},
    removeAttribute(name){delete attributes[name];},
    get attributes(){return attributes;},
    classList:{add(name){classes.add(name);},remove(name){classes.delete(name);},contains(name){return classes.has(name);}},
    addEventListener(event,callback){listeners[event]=callback;},
    href:null,target:null,rel:null,
    click(){listeners.click?.({preventDefault(){throw Error("native casino navigation must not be prevented");}});}
  };
}
function run(seed,stored="",override=null){
  const links=[makeElement("casinoRegisterUrl"),makeElement("lineContactUrl"),makeElement("lineContactUrl"),makeElement("dgTrialUrl")];
  const clicks={};
  const close={addEventListener(name,callback){clicks.close=callback;}};
  const dismiss={addEventListener(name,callback){clicks.dismiss=callback;}};
  const dialog={open:false,showCount:0,showModal(){this.open=true;this.showCount++;},close(){this.open=false;}};
  const saved=new Map();
  if(stored)saved.set("bbb-last-casino-link-v1",stored);
  const storage={getItem(key){return saved.get(key)||null;},setItem(key,value){saved.set(key,value);}};
  const document={
    querySelectorAll(selector){assert.equal(selector,"[data-bbb-link]");return links;},
    getElementById(id){
      if(id==="casinoAccountDialog")return dialog;
      if(id==="casinoAccountClose")return close;
      if(id==="casinoAccountDismiss")return dismiss;
      return null;
    }
  };
  const window={crypto:{getRandomValues(data){data[0]=seed;return data;}},sessionStorage:storage,BBB_SITE_LINKS:null};
  const ctx={window,document,URL,Uint32Array,Math,console};
  vm.runInNewContext(siteJs,ctx,{filename:"site-links.js"});
  if(override)window.BBB_SITE_LINKS=override;
  vm.runInNewContext(binderJs,ctx,{filename:"outbound-links.js"});
  return {links,dialog,storage,clicks};
}
for(let seed=0;seed<expected.length;seed++){
  const result=run(seed);
  assert.equal(result.links[0].href,expected[seed],"every casino is reachable");
  assert.equal(result.links[0].target,"_blank");
  assert(result.links[0].rel.includes("noopener"));
  assert(result.links[0].rel.includes("noreferrer"));
  assert.equal(result.links[1].href,"https://lin.ee/6vkiP3b");
  assert.equal(result.links[2].href,"https://lin.ee/6vkiP3b");
  assert.equal(result.links[3].href,"https://dg66.net/Opengame/freegame/");
  assert.equal(result.dialog.open,false);
  result.links[0].click();
  assert.equal(result.dialog.showCount,1);
  assert.equal(result.storage.getItem("bbb-last-casino-link-v1"),expected[seed]);
  result.clicks.close();
  assert.equal(result.dialog.open,false);
  result.links[0].click();
  assert.notEqual(result.links[0].href,expected[seed],
    "same page must reselect on every click, not stay at original casino");
  assert.equal(result.storage.getItem("bbb-last-casino-link-v1"),result.links[0].href,
    "session must track latest clicked destination");
  result.clicks.dismiss();
  assert.equal(result.dialog.open,false);
}
const next=run(0,expected[0]);
assert.notEqual(next.links[0].href,expected[0],"immediate repeat should be avoided");
const invalid=run(0,"",{casinoRegisterUrls:["javascript:alert(1)","http://not-allowed.test"],lineContactUrl:"javascript:alert(1)",dgTrialUrl:"http://not-allowed.test"});
assert.equal(invalid.links[0].attributes["aria-disabled"],"true");
assert.equal(invalid.links[1].attributes["aria-disabled"],"true");
const indexHtml=fs.readFileSync(new URL("./index.html",import.meta.url),"utf8");
const trialClient=fs.readFileSync(new URL("./trial-client.js",import.meta.url),"utf8");
assert(indexHtml.includes('id="dgDirectTrialLink"'),"independent DG trial button must be visible before backend authorization");
assert(indexHtml.includes('src="bankroll-trial.js?v=20261115-accuracy-trigger-v3"'));
assert(indexHtml.includes('src="trial-client.js?v=20261115-accuracy-trigger-v3"'));
assert(indexHtml.includes('src="trial-hit-rate.js?v=20261115-accuracy-trigger-v1"'));
assert(indexHtml.includes('id="bbbBankrollDialog"'),"trial must show bankroll setup before starting");
assert(!trialClient.includes('window.open(url.href,"_blank","noopener,noreferrer")'),"BBB trial must never launch DG automatically");
assert(!trialClient.includes("openDgOnUserClick();"),"remove old DG side effect");
const clickPos=trialClient.indexOf('startBtn.addEventListener("click",()=>{');
const setupPos=trialClient.indexOf("  if(!bankroll.consumeConfirmation()){",clickPos);
const startPos=trialClient.indexOf("  startOrResume();",clickPos);
assert(clickPos>=0&&setupPos>clickPos&&startPos>setupPos,"require bankroll before the BBB trial API");
console.log(JSON.stringify({ok:true,checks:["five_equal_weight_options","exact_affiliate_urls","line_transfer","new_tab","account_dialog","per_click_reselection","avoids_last_site","https_guard","dg_is_independent","trial_requires_bankroll"]}));

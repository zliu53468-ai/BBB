// First-visit no-code landing and tracked return CTA; no browser/network required.
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
const index=fs.readFileSync(new URL("./index.html",import.meta.url),"utf8");
const code=fs.readFileSync(new URL("./visitor-flow.js",import.meta.url),"utf8");
const end=fs.readFileSync(new URL("./trial-ended.html",import.meta.url),"utf8");
const outbound=fs.readFileSync(new URL("./outbound-links.js",import.meta.url),"utf8");
assert(/id="licenseMemberPanel"[^>]*hidden/.test(index),"member code form must start hidden");
assert(/id="returnVisitorCard"[^>]*hidden/.test(index),"return CTA must start hidden");
for(const id of ["trialStartBtn","dgDirectTrialLink","licenseForm","licenseMessage","predictionApp","licenseGate"]){
  assert(index.includes('id="'+id+'"'),"required UI element "+id);
}
assert(index.indexOf('id="dgDirectTrialLink"')<index.indexOf('id="licenseForm"'),"DG must be primary entry");
assert(index.indexOf('id="trialStartBtn"')<index.indexOf('id="licenseForm"'),"free trial must be primary entry");
assert(index.includes("https://bgsv1.xyz/"),"keep canonical SEO URL");
assert(end.includes('id="casinoNextHint"'),"next casino return hint exists");
assert(outbound.includes("window.setTimeout(prepareNextCasino,0)"),"next link refreshed automatically after click");
assert(outbound.includes('markFunnel("casino")'),"casino click persists visitor funnel");
assert(outbound.includes('markFunnel("line")'),"LINE click persists visitor funnel");
assert(outbound.includes('event?.persisted&&lastOpenedInPage'),"BFCache return refresh");
const now=Date.now();
function run(action=null,search=""){
  const nodes=Object.fromEntries(["returnVisitorCard","licenseMemberPanel","memberAccessToggle","returnVisitorReason"].map(id=>[id,{
    hidden:true,textContent:"",setAttribute(name,value){this[name]=value;},
    addEventListener(name,cb){this[name]=cb;},
    scrollIntoView(){this.scrolled=true;}
  }]));
  const key="bbb-funnel-last-action-v1";
  const stored=new Map();
  if(action)stored.set(key,JSON.stringify({action,at:now}));
  const storage={getItem(key){return stored.get(key)||null;}};
  const window={addEventListener(name,cb){this[name]=cb;}};
  const document={getElementById(id){return nodes[id]||null;}};
  vm.runInNewContext(code,{window,document,location:{search},localStorage:storage,sessionStorage:storage,URLSearchParams,Date}, {filename:"visitor-flow.js"});
  return nodes;
}
const first=run();
assert.equal(first.returnVisitorCard.hidden,true,"first visitor should not see code offer");
assert.equal(first.licenseMemberPanel.hidden,true,"first visitor should not see code form");
const afterCasino=run("casino");
assert.equal(afterCasino.returnVisitorCard.hidden,false,"casino visitor gets claim CTA");
assert.equal(afterCasino.licenseMemberPanel.hidden,true,"code form stays hidden until explicitly requested");
afterCasino.memberAccessToggle.click();
assert.equal(afterCasino.licenseMemberPanel.hidden,false,"registered visitor can show existing code form");
const afterLine=run("line");
assert.equal(afterLine.returnVisitorCard.hidden,false,"LINE visitor gets claim CTA");
const directMember=run(null,"?license=1");
assert.equal(directMember.licenseMemberPanel.hidden,false,"member login deep-link remains available");
console.log(JSON.stringify({ok:true,checks:["dg_first","trial_first","no_code_first_visit","funnel_after_casino","funnel_after_line","explicit_member_form","member_deep_link","return_rotates_no_reload"]}));

import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
const read=name=>fs.readFileSync(new URL("./"+name,import.meta.url),"utf8");
const home=read("index.html"),exit=read("trial-ended.html"),css=read("ui-three-tabs.css");
const tabs=read("homepage-tabs.js"),zoom=read("feedback-lightbox.js");
const trial=read("trial-client.js"),tracking=read("trial-hit-rate.js");
for(const id of ["bbbTabBtnExperience","bbbTabBtnFeedback","bbbTabBtnLearn",
 "bbbTabExperience","bbbTabFeedback","bbbTabLearn"])assert(home.includes('id="'+id+'"'),"tab id "+id);
assert((home.match(/<button[^>]*id="bbbTabBtn(?:Experience|Feedback|Learn)"[^>]*role="tab"/g)||[]).length===3,"three legacy landing tabs preserved");
assert((home.match(/<div[^>]*id="bbbTab(?:Experience|Feedback|Learn)"[^>]*role="tabpanel"/g)||[]).length===3,"three legacy landing panels preserved");
assert((home.match(/<button[^>]*id="quantTabBtn(?:Probability|Kelly|Simulation)"[^>]*role="tab"/g)||[]).length===3,"three new quant tabs coexist without replacing legacy tabs");
assert((home.match(/<div[^>]*id="quantTab(?:Probability|Kelly|Simulation)"[^>]*role="tabpanel"/g)||[]).length===3,"three new quant panels are source-rendered");
assert(home.includes('id="bbbTabFeedback" class="bbb-landing-tab-panel" role="tabpanel" aria-labelledby="bbbTabBtnFeedback" tabindex="0" hidden'),"feedback collapsed initially");
assert(home.includes('id="bbbTabLearn" class="bbb-landing-tab-panel" role="tabpanel" aria-labelledby="bbbTabBtnLearn" tabindex="0" hidden'),"guides collapsed initially");
assert(home.includes('id="trialStartBtn"')&&home.includes('href="partner-entry.html"')&&!home.includes('id="dgDirectTrialLink"'),"trial and partner entry preserved without external free demo");
assert(home.includes('id="licenseForm"')&&home.includes('id="bbbBankrollDialog"'),"license and capital input preserved");
assert(home.includes('id="bbbTrialBankrollPanel"'),"money tracking DOM retained without exposing huge UI");
assert(css.includes("#predictionApp #bbbTrialBankrollPanel{display:none!important}"),"huge money panel hidden");
assert(home.includes('id="bbbTrialAccuracyHits"')&&home.includes('id="bbbTrialAccuracyValid"')&&home.includes('id="bbbTrialAccuracyRate"'),"actual trial hit-rate counters preserved");
assert(home.includes('提前結束條件：至少 10 次、命中率 ≥ 60%'),"original threshold preserved for regression");
assert(tracking.includes('window.location.replace("./trial-ended.html?reason=accuracy60")'),"60 percent expiry stays operational");
assert(trial.includes("window.BBB_TRIAL_ACCURACY?.onTrialMounted"),"tracker startup remains");
assert(home.includes('class="metrics bbb-hide-diagnostics"'),"technical four-card section still has original IDs");
for(const id of ["confidence","regime","roadAuxStatus","strength"])assert(home.includes('id="'+id+'"'),"EV model bindings remain "+id);
assert(css.includes("#predictionApp .prediction-card .metrics.bbb-hide-diagnostics{display:none!important}"),"EV cards hidden visually");
assert(!exit.includes('class="casino-rotation-explain"'),"requested exit explainer removed");
assert(exit.includes('data-bbb-link="casinoRegisterUrl"')&&exit.includes('data-bbb-link="lineContactUrl"'),"original routes untouched");
assert(exit.includes('id="casinoAccountDialog"'),"original account modal untouched");
assert((home.match(/class="member-feedback-card"/g)||[]).length===5,"five screenshots retained");
assert((home.match(/class="bbb-feedback-expand"/g)||[]).length===5,"five clickable zoom surfaces");
for(let i=1;i<=5;i++)assert(home.includes('data-feedback-index="'+i+'"'),"feedback index "+i);
assert(home.includes('id="bbbFeedbackLightbox"')&&home.includes('id="bbbFeedbackZoomViewport"'),"accessible screenshot zoom dialog");
for(let i=1;i<=5;i++)assert(home.includes('src="assets/member-feedback-0'+i+'.webp"'),"original full-resolution photo "+i);
assert(css.includes("@media(max-width:640px)"),"responsive phone styles");
assert(css.includes("@media(min-width:1024px)"),"responsive desktop styles");

// Test tabs as an actual isolated event-driven browser module.
function fake(id,selected=false){
 const listeners={};
 return{
  id,hidden:false,tabIndex:selected?0:-1,focusCount:0,
  attrs:{"aria-selected":String(selected)},
  addEventListener(key,fn){(listeners[key]??=[]).push(fn);},
  setAttribute(key,value){this.attrs[key]=value;},
  getAttribute(key){return this.attrs[key];},
  focus(){this.focusCount++;},
  fire(key,props={}){for(const fn of listeners[key]||[])fn({preventDefault(){this.prevented=true;},...props});}
 };
}
const ids=["bbbTabBtnExperience","bbbTabBtnFeedback","bbbTabBtnLearn","bbbTabExperience","bbbTabFeedback","bbbTabLearn"];
const els=Object.fromEntries(ids.map((id,i)=>[id,fake(id,i===0)]));
vm.runInNewContext(tabs,{document:{getElementById(id){return els[id]||null;}}});
const [a,b,c]=ids.slice(0,3).map(x=>els[x]),[pa,pb,pc]=ids.slice(3).map(x=>els[x]);
assert.equal(pa.hidden,false);assert.equal(pb.hidden,true);assert.equal(pc.hidden,true);
b.fire("click");
assert.equal(pa.hidden,true);assert.equal(pb.hidden,false);assert.equal(pc.hidden,true);
assert.equal(b.attrs["aria-selected"],"true");assert.equal(a.tabIndex,-1);
b.fire("keydown",{key:"ArrowRight"});
assert.equal(pc.hidden,false);assert.equal(c.focusCount,1);
c.fire("keydown",{key:"Home"});
assert.equal(pa.hidden,false);assert.equal(a.focusCount,1);
a.fire("keydown",{key:"ArrowLeft"});
assert.equal(pc.hidden,false);assert.equal(c.focusCount,2);

// Test clickable gallery, wraparound and modal close.
const screens=[1,2,3,4,5].map(i=>{const el=fake("feedback"+i);el.dataset={feedbackIndex:String(i)};return el;});
const modal=fake("bbbFeedbackLightbox");modal.open=false;
modal.showModal=function(){this.open=true};
modal.close=function(){this.open=false;this.fire("close");};
const view=fake("bbbFeedbackZoomViewport");view.className="";
const image=fake("bbbFeedbackZoomImg");image.src="";
const original=fake("bbbFeedbackZoomOriginal");original.href="";
const count=fake("bbbFeedbackZoomIndex");count.textContent="";
const close=fake("bbbFeedbackZoomClose");
const prev=fake("bbbFeedbackZoomPrev");
const next=fake("bbbFeedbackZoomNext");
const doc={getElementById(id){return ({
 bbbFeedbackLightbox:modal,bbbFeedbackZoomViewport:view,bbbFeedbackZoomImg:image,bbbFeedbackZoomOriginal:original,bbbFeedbackZoomIndex:count,
 bbbFeedbackZoomClose:close,bbbFeedbackZoomPrev:prev,bbbFeedbackZoomNext:next
})[id]||null;},querySelectorAll(selector){assert.equal(selector,"[data-feedback-index]");return screens;}};
vm.runInNewContext(zoom,{document:doc});
screens[0].fire("click");
assert.equal(modal.open,true);
assert.equal(count.textContent,"1 / 5","first click opens correct image");
assert.equal(image.src,"assets/member-feedback-01.webp","zoom loads first native-resolution image");
assert.equal(original.href,"assets/member-feedback-01.webp","original-size link points to native asset");
next.fire("click");assert.equal(count.textContent,"2 / 5");assert.equal(image.src,"assets/member-feedback-02.webp");
prev.fire("click");assert.equal(count.textContent,"1 / 5");
prev.fire("click");assert.equal(count.textContent,"5 / 5","prev wraps at beginning");
next.fire("click");assert.equal(count.textContent,"1 / 5","next wraps at end");
close.fire("click");assert.equal(modal.open,false);
assert.equal(screens[0].focusCount>0,true,"focus restored after closing");
console.log(JSON.stringify({ok:true,checks:["three_tabs_with_keyboard","responsive_views","five_zoomable_feedbacks","modal_accessibility","correct_image_index","arrow_wraparound","preserved_auth_and_trial","60_percent_tracking","hidden_money_and_EV_panels","exit_text_removed"]}));

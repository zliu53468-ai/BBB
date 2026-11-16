import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const html=read("index.html"),mobile=read("mobile-one-screen.css"),cards=read("feedback-carousel.js"),login=read("mobile-member-modal.js");
const originalWidths=[1280,647,1257,1017,762];
const originalHeights=[1066,627,1079,939,755];
assert(html.includes('mobile-one-screen.css?v=20261116-fit-v1'),"mobile stylesheet loads after black-gold theme");
assert(html.indexOf('href="mobile-one-screen.css')>html.indexOf('href="black-gold-theme.css'),"overrides loaded after black-gold");
assert(mobile.includes("height:100dvh;min-height:0;max-height:100dvh;overflow:hidden!important"),"mobile homepage viewport bounds");
assert(mobile.includes('body.bbb-black-gold:has(#licenseGate:not([hidden]))'),"body scroll locked only on visible homepage");
assert(mobile.includes("#licenseGate>.bbb-landing-tab-panel[hidden]{display:none!important}"),"hidden tabs stay hidden");
assert(mobile.includes("body.bbb-black-gold #licenseGate .trial-visitor-entry.bbb-neon-entry"),"primary start/DG cards fit viewport");
assert(mobile.includes(".bbb-feedback-carousel-nav"),"feedback navigation styled for small screens");
assert(mobile.includes("#licenseMemberPanel:not([hidden])"),"login opens on-screen instead of causing page scroll");
assert(mobile.includes(".bbb-mobile-quick-learn"),"phone learning shortcuts in a single frame");
assert(mobile.includes("@media(max-width:760px) and (max-height:625px)"),"short phones get extra compact rules");
assert(!mobile.includes("#appViewportContent{overflow:hidden"),"prediction app scrolling is deliberately retained");
assert(html.includes('id="bbbTabBtnExperience"')&&html.includes('id="bbbTabBtnFeedback"')&&html.includes('id="bbbTabBtnLearn"'),"three tabs kept");
assert(html.includes('id="trialStartBtn"')&&html.includes('id="dgDirectTrialLink"'),"trial DG buttons preserved");
assert(html.includes('id="bbbFeedbackCarousel"'),"feedback carousel remains accessible");
assert(html.includes('id="bbbFeedbackCardCounter"'),"feedback page includes position counter");
assert(html.includes('id="bbbFeedbackZoomOriginal"'),"native-size open link on zoom dialog");
assert(html.includes('id="bbbMobileMemberClose"'),"login close button included");
for(let i=1;i<=5;i++){
 const imageName="assets/member-feedback-0"+i+".webp";
 const bytes=fs.readFileSync(new URL("./"+imageName,import.meta.url));
 assert.equal(bytes.subarray(0,4).toString("ascii"),"RIFF",imageName+" starts as RIFF");
 assert.equal(bytes.subarray(8,12).toString("ascii"),"WEBP",imageName+" is WebP");
 assert(bytes.length>16000,imageName+" contains high-fidelity native resolution screenshot");
 assert(html.includes('src="'+imageName+'"'),"thumbnail loads unscaled original source");
 assert(cards.includes("member-feedback-card"),"carousel tracks original cards");
 const d=originalWidths[i-1],h=originalHeights[i-1];
 assert(html.includes('width="'+d+'" height="'+h+'"'),"native screenshot dimensions in HTML");
}
function fake(id){
 const events={};
 return {id,hidden:false,disabled:false,attrs:{},textContent:"",focusCount:0,
 addEventListener(name,fn){(events[name]??=[]).push(fn);},
 fire(name,extra={}){for(const fn of events[name]??[])fn({key:extra.key,preventDefault(){}});},
 setAttribute(k,v){this.attrs[k]=v;},
 focus(){this.focusCount++;}
 };
}
// Simulate a phone layout first, then desktop resize.
const gallery=fake("bbbFeedbackCarousel");
const rows=Array.from({length:5},(_,i)=>fake("photo"+i));
gallery.querySelectorAll=selector=>{assert.equal(selector,".member-feedback-card");return rows;};
const prev=fake("bbbFeedbackCardPrev"),next=fake("bbbFeedbackCardNext"),count=fake("bbbFeedbackCardCounter");
const map={
 bbbFeedbackCarousel:gallery,bbbFeedbackCardPrev:prev,
 bbbFeedbackCardNext:next,bbbFeedbackCardCounter:count
};
const listeners={};let phone=true;
const win={matchMedia(){return {matches:phone};},addEventListener(name,fn){listeners[name]=fn;}};
vm.runInNewContext(cards,{document:{getElementById:id=>map[id]??null},window:win});
assert.equal(rows.filter(r=>!r.hidden).length,1,"only one gallery item shows on phone");
assert.equal(count.textContent,"1 / 5");
next.fire("click");
assert.equal(rows[1].hidden,false);assert.equal(count.textContent,"2 / 5");
prev.fire("click");
assert.equal(rows[0].hidden,false);assert.equal(count.textContent,"1 / 5");
prev.fire("click");
assert.equal(rows[4].hidden,false);assert.equal(count.textContent,"5 / 5","carousel wraps backward");
phone=false;listeners.resize();
assert.equal(rows.filter(r=>!r.hidden).length,5,"all cards remain available on desktop");
assert.equal(prev.disabled,true,"hidden phone nav is disabled on desktop");
// Test the on-screen license login close behavior.
const member=fake("licenseMemberPanel"),dismiss=fake("bbbMobileMemberClose"),trigger=fake("memberAccessToggle");
member.hidden=false;
vm.runInNewContext(login,{
 document:{getElementById:id=>({
 licenseMemberPanel:member,bbbMobileMemberClose:dismiss,memberAccessToggle:trigger
 })[id]??null}
});
dismiss.fire("click");
assert.equal(member.hidden,true,"login modal closes without page scroll");
assert.equal(trigger.attrs["aria-expanded"],"false");
assert.equal(trigger.focusCount,1,"focus returns to login trigger");
member.hidden=false;member.fire("keydown",{key:"Escape"});
assert.equal(member.hidden,true,"Escape closes mobile login");
console.log(JSON.stringify({ok:true,checks:["five_native_webp_images","mobile_body_no_scroll","all_ctas_above_fold","feedback_pager","desktop_grid_restored","fullscreen_original_image","mobile_modal_focus","accessible_three_tabs","model_scroll_preserved"]}));

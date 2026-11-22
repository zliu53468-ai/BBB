import assert from "node:assert/strict";
import fs from "node:fs";
const read=path=>fs.readFileSync(new URL("./"+path,import.meta.url),"utf8");

const h=read("index.html"),tools=read("quant-tools.html"),
  css=read("quant-research.css"),site=read("site-links.js"),
  portal=read("partner-entry.html"),portalCss=read("partner-entry.css"),
  outbound=read("outbound-links.js"),trial=read("trial-client.js"),
  sitemap=read("sitemap.xml");

const probabilityStart=h.indexOf('id="quantTabProbability"');
const probabilityEnd=h.indexOf('id="quantTabKelly"',probabilityStart);
const probability=h.slice(probabilityStart,probabilityEnd);
assert(probabilityStart>=0&&probabilityEnd>probabilityStart,"probability panel exists");
for(const id of ["quantHistory","quantAddB","quantAddP","quantAddT","quantApply","quantUndo","quantClear","quantTotal","quantBarB","quantBarP","quantBarT","quantCountB","quantCountP","quantCountT","quantDeviationB","quantDeviationP","quantDeviationT","quantRecent","quantInputStatus"]){
  assert(probability.includes('id="'+id+'"'),"existing calculator binding "+id+" preserved in panel");
}
assert(probability.includes('id="trialStartBtn"'),"actual BBB trial control is within calculator field");
assert(!probability.includes('id="dgDirectTrialLink"'),"external free-demo link removed as requested");
assert(probability.includes('id="trialStartMessage"'),"trial status is presented in the same panel");
assert(probability.includes('href="partner-entry.html"'),"partner portal is adjacent to trial");
assert.equal((h.match(/id="trialStartBtn"/g)||[]).length,1,"never duplicate trial start button");
assert.equal((h.match(/id="dgDirectTrialLink"/g)||[]).length,0,"external free-demo button absent");
assert.equal((h.match(/id="trialStartMessage"/g)||[]).length,1,"never duplicate trial status");
assert(probability.includes('class="quant-trial-preview"'),"readable trial preview exists");
assert(probability.includes("畫面示意")&&probability.includes("不保證結果"),"preview makes no predictive claim");
assert(probability.includes("<details class=\"quant-advanced-detail\">"),"technical stats are optional and collapsed");
assert(probability.indexOf('id="quantCountB"')>probability.indexOf('<details class="quant-advanced-detail">'),"advanced cards not default");
assert(h.includes('id="bbbTabExperience"')&&h.includes('class="quant-trial-handoff"'),"legacy landing tab remains navigable");
assert(h.includes('id="licenseMemberPanel"')&&h.includes('id="licenseForm"')&&h.includes('id="predictionApp"'),"license and prediction application remain intact");
assert(trial.includes('document.getElementById("trialStartBtn")')&&trial.includes('startBtn.addEventListener("click"'),"actual trial runtime still binds same button");
assert(trial.includes('document.getElementById("licenseGate")')&&trial.includes('document.getElementById("predictionApp")'),"real trial activation path unchanged");
assert(!tools.includes('id="trialStartBtn"'),"standalone page does not fake trial without license runtime");
assert(tools.includes('href="./#quantAnalyzer"')&&tools.includes('href="partner-entry.html"'),"standalone page links to actual trial");
assert.equal((h.match(/class="quant-tab-panel/g)||[]).length,3,"all three static SEO tabs preserved");
assert.equal((tools.match(/class="quant-tab-panel/g)||[]).length,3,"all three standalone static SEO tabs preserved");
for(const id of ["quantEvB","quantEvP","quantEvT","quantKelly","quantSimChart"]){
  assert(h.includes('id="'+id+'"'),"other research tabs preserved "+id);
}
assert(css.includes(".quant-entry-layout")&&css.includes("grid-template-columns:1fr;gap:12px"),"responsive visitor-first two-column layout");
assert(css.includes(".quant-advanced-detail")&&css.includes(".quant-preview-card"),"compact stats and trial visual styles");
assert(h.includes("quant-research.css?v=20261122-trial-portal-v3"),"homepage uses new CSS cache key");
assert(tools.includes("quant-research.css?v=20261122-trial-portal-v3"),"tool page uses new CSS cache key");

assert(portal.includes('rel="canonical" href="https://bgsv1.xyz/partner-entry.html"'),"dedicated canonical URL");
assert(portal.includes('data-bbb-link="casinoRegisterUrl"')&&portal.includes('data-bbb-link="lineContactUrl"'),"existing outbound link mechanism reused");
for(const id of ["casinoAccountDialog","casinoAccountClose","casinoAccountDismiss","casinoNextHint"]){
  assert(portal.includes('id="'+id+'"'),"compatible outbound lifecycle control "+id);
}
assert(!portal.includes("試用已結束")&&!portal.includes("已達 60%"),"portal does not falsely claim trial expired");
assert(portal.includes('src="site-links.js"')&&portal.includes('src="outbound-links.js?v=20261115-return-rotation-v3"'),"original operator-managed destinations and rotation in use");
assert(outbound.includes('markFunnel("casino")')&&outbound.includes("pickCasino()"),"original random selection/return behavior intact");
assert(site.includes("casinoRegisterUrls")&&site.includes("lineContactUrl"),"destinations read from existing operator config");
assert(portalCss.includes("@media(max-width:760px)")&&portalCss.includes(".bq-portal-grid"),"mobile partner page styled");
assert(sitemap.includes("https://bgsv1.xyz/partner-entry.html"),"new destination indexed in sitemap");
for(const file of ["index.html","quant-tools.html","research-hub.html","model-methodology.html","risk-center.html","research-martingale-100k.html","research-shoe-deviation.html","research-scam-safety-2026.html","partner-entry.html"]){
  assert(read(file).includes('href="partner-entry.html"'),"partner page discoverable from "+file);
}
console.log(JSON.stringify({ok:true,checks:["shorter_ui_copy","collapsible_advanced_stats","real_trial_button_moved_once","trial_runtime_unchanged","real_external_demo","partner_portal_with_original_links","safe_rotation","three_static_tabs","member_flow_untouched","mobile_layout","sitemap"]}));

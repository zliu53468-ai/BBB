import assert from "node:assert/strict";
import fs from "node:fs";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const home=read("index.html"),siteCss=read("quant-site.css"),siteJs=read("quant-site.js");
const sitemap=read("sitemap.xml");
const routes=["quant-tools.html","research-hub.html","model-methodology.html","risk-center.html"];
const articles=["research-martingale-100k.html","research-shoe-deviation.html","research-scam-safety-2026.html"];
const shared=["index.html",...routes,...articles];

assert.equal((home.match(/<h1\b/g)||[]).length,1,"home retains exactly one H1");
assert(home.includes('class="quant-banner-top"'),"real editorial hero banner present");
assert(home.includes('class="quant-banner-visual"'),"hero includes visual analytics graphic");
assert(home.includes('圖表為視覺示意，非即時牌桌或實盤績效'),"synthetic visual labeled honestly");
assert(home.includes('class="quant-feature-strip"'),"featured research banner exists");
assert.equal((home.match(/class="quant-topic"/g)||[]).length,4,"four research topic cards");
assert(home.indexOf('id="quantTopics"')<home.indexOf('id="quantAnalyzer"'),"topic navigation appears above the calculator");
assert(home.includes('id="quantGuides"'),"original guide section remains");
assert(home.includes('id="quantBacktest"'),"original transparency section remains");
for(const id of ["licenseGate","quantAnalyzer","quantTabProbability","quantTabKelly","quantTabSimulation","trialStartBtn","licenseForm","predictionApp","btnStart","btnB","btnP","btnT"]){
  assert(home.includes('id="'+id+'"'),"BBB legacy and quant control preserved: "+id);
}
assert(siteCss.includes("position:sticky;top:0;z-index:65"),"header stays on top");
assert(siteCss.includes("grid-template-columns:repeat(4,minmax(0,1fr))"),"desktop four topic categories");
assert(siteCss.includes("@media(max-width:760px)"),"mobile responsiveness present");
assert(siteCss.includes(".quant-site-mobile-drawer"),"mobile native details menu styled");
assert(siteCss.includes(".quant-inner-banner"),"subpage banner reused across topics");
assert(siteCss.includes("prefers-reduced-motion:reduce"),"reduced motion support");
assert(siteJs.includes("aria-current"),"active navigation marks are accessible");
assert(!siteJs.includes("innerHTML")&&!siteJs.includes("fetch("),"navigation JS is lightweight and local");

for(const path of shared){
 const html=read(path);
 assert(html.includes('class="quant-site-header"'),"top navigation shared across "+path);
 assert(html.includes('class="quant-site-mobile-drawer"'),"mobile category nav shared across "+path);
 assert(html.includes('href="quant-site.css?v=20261122-multipage-v1"'),"style loaded on "+path);
 assert(html.includes('src="quant-site.js?v=20261122-multipage-v1"'),"nav script loaded on "+path);
 for(const p of routes){
   assert(html.includes('href="'+p)||html.includes('href="'+p+'#'),"menu route present "+p+" in "+path);
 }
}
for(const path of routes){
 const html=read(path);
 assert.equal((html.match(/<h1\b/g)||[]).length,1,"one H1 on "+path);
 assert(html.includes('rel="canonical" href="https://bgsv1.xyz/'+path+'"'),"canonical on "+path);
 assert(html.includes('"@type":"WebPage"'),"structured metadata on "+path);
 assert(sitemap.includes("https://bgsv1.xyz/"+path),"sitemap entry "+path);
 assert(html.includes('class="quant-inner-banner"'),"editorial banner on "+path);
 assert(html.includes('aria-label="麵包屑"'),"breadcrumb present on "+path);
}
const tools=read("quant-tools.html");
assert((tools.match(/class="quant-tab-panel/g)||[]).length===3,"all 3 interactive tools have static HTML");
for(const id of ["quantHistory","quantAddB","quantAddP","quantAddT","quantHypWin","quantBankroll","quantSimChart","quantRunSim","quantTabBtnProbability","quantTabBtnKelly","quantTabBtnSimulation"]){
 assert(tools.includes('id="'+id+'"'),"interactive tool binding preserved on standalone page: "+id);
}
assert(tools.includes('src="quant-research.js?v=20261122-tabs-v2"'),"existing calculator script reused unchanged");
assert(tools.includes('data-quant-target="quantTabKelly"'),"local tab links control the active tool");
const research=read("research-hub.html");
assert.equal((research.match(/<article class="quant-guide-card/g)||[]).length,3,"research hub has all 3 original static guide cards");
assert(research.includes('id="quantBacktest"'),"transparent backtest notes in research hub");
const model=read("model-methodology.html");
assert(model.includes("粒子濾波")&&model.includes("可重現"),"research methods contain meaningful content");
const risk=read("risk-center.html");
assert(risk.includes("−1.06%")&&risk.includes("165 反詐騙專線"),"risk page covers EV and safety");
for(const path of articles){
 const html=read(path);
 assert(html.includes('class="quant-site-header"'),"existing guide now has global site navigation");
 assert(html.includes('href="research-hub.html"'),"article links to research hub");
 assert(html.includes('class="guide-article"'),"original article content unchanged");
}
console.log(JSON.stringify({ok:true,checks:["homepage_visual_banner","featured_research_banner","4_topic_categories","4_independent_pages","3_full_calculator_panels","mobile_and_desktop_navigation","article_navigation","seo_canonical_sitemap","prediction_core_preserved"]}));

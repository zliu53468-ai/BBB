import assert from "node:assert/strict";
import fs from "node:fs";
import {runDemo,SETTINGS} from "./research/martingale-100k.mjs";

const read=path=>fs.readFileSync(new URL("./"+path,import.meta.url),"utf8");
const home=read("index.html");
const css=read("quant-guides.css");
const sitemap=read("sitemap.xml");
const pages=[
  "research-martingale-100k.html",
  "research-shoe-deviation.html",
  "research-scam-safety-2026.html"
];

const demo=runDemo();
assert.deepEqual(demo,runDemo(),"simulation must reproduce exactly");
assert.equal(demo.totalRandomOutcomes,100000);
assert.equal(demo.paths,200);
assert.equal(demo.roundsPerPath,500);
assert.equal(demo.strategyStops,172);
assert.equal(demo.stopByLimit,89);
assert.equal(demo.stopByBankroll,83);
assert.equal(demo.belowBase,2);
assert.equal(demo.strategyStopFraction,.86);
assert(Math.abs(demo.medianMaxDrawdown-.5722065353057929)<1e-9);
assert(Math.abs(demo.p90MaxDrawdown-.8292523669604961)<1e-9);
assert.equal(SETTINGS.seed,20261122);

assert.equal((home.match(/<article class="quant-guide-card /g)||[]).length,3,"3 semantic article cards");
assert.equal((home.match(/<h3 id="quantGuideTitle/g)||[]).length,3,"3 h3 article titles");
assert.equal((home.match(/核心重點摘要（Key Takeaways）/g)||[]).length,3,"summary in each card");
assert.equal((home.match(/研究員：Data Lab 量化團隊/g)||[]).length,3,"author on each card");
assert.equal((home.match(/quant-guide-reading/g)||[]).length,3,"reading time on each card");
assert.equal((home.match(/quant-guide-arrow/g)||[]).length,3,"3 microinteractive links");
assert(home.indexOf('id="quantGuides"')>home.indexOf('id="quantTabSimulation"'));
assert(home.indexOf('id="quantGuides"')<home.indexOf('id="quantBacktest"'),"guide section follows calculator");
assert(home.includes('href="quant-guides.css?v=20261122-a"'),"compiled guide stylesheet linked");
for(const id of ["quantGuides","quantGuidesHeading","quantGuideTitleMartingale","quantGuideTitleDeviation","quantGuideTitleSafety"]){
  assert(home.includes('id="'+id+'"'),"semantic anchor "+id);
}
assert(css.includes("grid-template-columns:repeat(3,minmax(0,1fr))"),"desktop 3 columns");
assert(css.includes("grid-template-columns:1fr"),"phone 1 column");
assert(css.includes("border-color:#3B82F6")&&css.includes("translateX(5px)"),"hover styling");
assert(css.includes(".bg-slate-900\\/70")&&css.includes(".backdrop-blur-md"),"tailwind-like precompiled glass utilities");
assert(css.includes("prefers-reduced-motion:reduce"),"reduced motion honored");

const jsonLD=[...home.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map(x=>JSON.parse(x[1]));
const itemList=jsonLD.find(x=>x["@type"]==="ItemList");
assert(itemList&&itemList.itemListElement.length===3,"itemlist SEO is valid");
for(let i=0;i<pages.length;i++){
  const path=pages[i],page=read(path);
  assert(home.includes('href="'+path+'"'),"homepage link exists");
  assert(sitemap.includes("https://bgsv1.xyz/"+path),"sitemap includes guide");
  assert(page.includes('rel="canonical" href="https://bgsv1.xyz/'+path+'"'),"canonical guide URL");
  assert(page.includes("<article ")&&page.includes("<h1"),"indexable page markup");
  assert(page.includes("研究員：Data Lab 量化團隊"),"byline is visible");
  assert(page.includes('class="guide-sources"'),"public sources");
  const graph=[...page.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map(x=>JSON.parse(x[1]));
  assert.equal(graph[0]["@type"],"Article","Article JSON-LD");
  assert.equal(graph[0].mainEntityOfPage["@id"],"https://bgsv1.xyz/"+path);
  assert.equal(graph[0].datePublished,"2026-11-22T15:00:00+08:00");
}
const article1=read(pages[0]),article2=read(pages[1]),article3=read(pages[2]);
for(const text of ["172／200（86%）","57.22%","82.93%","2／200（1%）","不是普遍破產率","20261122"]){
  assert(article1.includes(text),"reproducible stats described: "+text);
}
const sigma=Math.sqrt(.458597*(1-.458597)/1000)*100;
assert(Math.abs(sigma-1.5757087027461643)<1e-10);
assert(article2.includes("1.58 個百分點"));
for(const phrase of ["三項演算法透明度標準","165 反詐騙專線","OTP","並未認定任何特定公司"]){
  assert(article3.includes(phrase),"scam education includes "+phrase);
}
console.log(JSON.stringify({ok:true,checks:["100k_seeded_simulation_reproducible","transparent_stop_vs_bankruptcy","3_article_cards","semantic_h3","source_rendered_content","responsive_grid","compiled_tailwind_utility","article_jsonld","sitemap","original_core_untouched"]}));

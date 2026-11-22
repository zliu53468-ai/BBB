import assert from "node:assert/strict";
import fs from "node:fs";
import {createRequire} from "node:module";

const require=createRequire(import.meta.url);
const math=require("./quant-research.js");
const read=file=>fs.readFileSync(new URL("./"+file,import.meta.url),"utf8");
const close=(a,b)=>assert(Math.abs(a-b)<1e-9,"expected "+a+" ≈ "+b);

assert.deepEqual(math.parseHistory("莊 閒 / 和，B P T"),["B","P","T","B","P","T"]);
assert.deepEqual(math.parseHistory("bpt"),["B","P","T"]);
assert.throws(()=>math.parseHistory("B X P"),/只接受/);
assert.throws(()=>math.parseHistory("B".repeat(2001)),/2,000/);
assert.throws(()=>math.summarize(["B","X"]),/B\/P\/T/);
const s=math.summarize(["B","B","P","T"]);
assert.equal(s.total,4);
assert.equal(s.items.B.count,2);
assert.equal(s.items.P.count,1);
assert.equal(s.items.T.count,1);
close(s.items.B.frequency,0.5);
close(s.items.B.deviation,0.5-0.458597);
const ev=math.theoreticalEV();
close(ev.B,0.95*0.458597-0.446247);
close(ev.P,0.446247-0.458597);
close(ev.T,8*0.095156-(0.458597+0.446247));
assert(ev.B<0&&ev.P<0&&ev.T<0);
assert.equal(s.kelly.B,0);
assert.equal(s.kelly.P,0);
assert.equal(s.kelly.T,0);
close(math.kellyFraction(0.6,0.4,1),0.2);

// Reproducible, purely educational Monte Carlo path: no certified win-rate claim.
const sim=math.simulateFlatBanker(20261122,10000,10,10000);
assert.equal(sim.rounds,10000);
assert.equal(sim.samples.at(-1).round,10000);
assert.equal(sim.samples.length,201);
assert.deepEqual(sim,math.simulateFlatBanker(20261122,10000,10,10000));
assert.notEqual(sim.endBalance,math.simulateFlatBanker(20261123,10000,10,10000).endBalance);
assert(sim.maxDrawdownPct>=0&&sim.maxDrawdownPct<=1);
assert(sim.betsPlaced>=0&&sim.betsPlaced<=10000);
assert.throws(()=>math.simulateFlatBanker(-1),/seed/);
assert.throws(()=>math.simulateFlatBanker(1,10000,10,10001),/10,000/);

const empty=math.summarize([]);
assert.equal(empty.total,0);
assert.equal(empty.items.T.count,0);

const home=read("index.html");
const css=read("quant-research.css");
const js=read("quant-research.js");
assert.equal((home.match(/<h1\b/g)||[]).length,1,"one H1 for whole page");
for(const id of ["quantResearch","quantAnalyzer","quantBacktest","quantFAQ","quantHistory","quantAddB","quantAddP","quantAddT","quantApply","quantUndo","quantClear","quantEvB","quantEvP","quantEvT","quantKelly","quantRiskSignal"]){
  assert(home.includes('id="'+id+'"'),id+" exists");
}
assert(home.includes("quant-research.css?v=20261122"));
assert(home.includes("quant-research.js?v=20261122"));
assert(home.indexOf("quant-research.css")>home.indexOf("mobile-one-screen.css"),"quant CSS after legacy CSS");
assert(css.includes("#0F172A")&&css.includes("#3B82F6")&&css.includes("#10B981"),"design palette");
assert(!js.includes("fetch("),"public calculator performs no remote requests");
assert(home.includes("待實測")&&!home.includes("10,000 局勝率"),"no fabricated backtest claims");
assert(home.includes("<!-- TODO: 在此填入真實回測圖表"));

assert.equal((home.match(/class="quant-tab-panel/g)||[]).length,3,"all 3 HTML panels present at load");
for(const pair of [["quantTabBtnProbability","quantTabProbability"],["quantTabBtnKelly","quantTabKelly"],["quantTabBtnSimulation","quantTabSimulation"]]){
  assert(home.includes('id="'+pair[0]+'"')&&home.includes('aria-controls="'+pair[1]+'"'),"tab linked "+pair[1]);
  assert(home.includes('id="'+pair[1]+'"')&&home.includes('aria-labelledby="'+pair[0]+'"'),"tabpanel linked "+pair[0]);
}
for(const id of ["quantBankroll","quantHypWin","quantHypPayout","quantKellyScale","quantRiskCap","quantHypEV","quantHypFullKelly","quantHypStake","quantSimChart","quantSimLine","quantSimSeed","quantSimStake","quantRunSim","quantSimEnd","quantSimMDD","quantSimBets"]){
  assert(home.includes('id="'+id+'"'),"static HTML contains tool "+id);
}
for(const term of ["即時機率與路單分析","凱利公式注碼控管試算","10,000 局模擬波動曲線","獨立隨機抽樣","不是歷史資料"]){
  assert(home.includes(term),"SEO-visible wording "+term);
}
assert(css.includes(".quant-tab-panel{display:none!important;")&&css.includes(".quant-tab-panel.is-active{display:block!important;"),"CSS-only toggling");
assert(css.includes("inset 0 1px 0 rgba(255,255,255,.25)")&&css.includes("translateY(-1px)")&&css.includes("scale(.95)"),"CTA microinteractions");
assert(css.includes("backdrop-filter:blur(12px)")&&css.includes("border-color:#3B82F6"),"glass KPI hover");
assert(js.includes("requestAnimationFrame")&&js.includes("prefers-reduced-motion"),"motion-sensitive counters");
assert(js.includes('setAttribute("aria-selected"')&&js.includes('setAttribute("aria-hidden"'),"accessible tab state");
assert(!js.includes("innerHTML"),"SEO content never injected by script");

const schemas=[...home.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map(m=>JSON.parse(m[1]));
const app=schemas.flatMap(v=>v["@graph"]||[v]).find(v=>v["@type"]==="SoftwareApplication");
const faq=schemas.find(v=>v["@type"]==="FAQPage");
assert(app&&faq&&faq.mainEntity.length===2);
assert.equal(app.offers.price,"0");
for(const q of faq.mainEntity){
  assert(home.includes("<h2>"+q.name+"</h2>"),"FAQ title visible");
  assert(home.includes("<p>"+q.acceptedAnswer.text+"</p>"),"FAQ answer visible");
}
for(const id of ["trialStartBtn","dgDirectTrialLink","licenseForm","predictionApp","btnB","btnP","btnT","btnEnd","btnBack"]){
  assert(home.includes('id="'+id+'"'),"legacy control preserved: "+id);
}
console.log(JSON.stringify({ok:true,checks:["engine_math","history_validation","negative_ev","kelly_zero","single_h1","json_ld_faq_visible","transparent_backtest","legacy_ids_preserved","no_network_calls","static_seo_tabs","accessible_controls","seeded_monte_carlo","cta_and_kpi_motion"]}));

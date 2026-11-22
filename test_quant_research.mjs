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
console.log(JSON.stringify({ok:true,checks:["engine_math","history_validation","negative_ev","kelly_zero","single_h1","json_ld_faq_visible","transparent_backtest","legacy_ids_preserved","no_network_calls"]}));

import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {createRequire} from "node:module";
const require=createRequire(import.meta.url),math=require("./quant-research.js");
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const home=read("index.html"),tools=read("quant-tools.html");
const css=read("bq-studio.css"),code=read("bq-studio.js");

for(const page of [home,tools]){
  assert.equal((page.match(/<h1\b/g)||[]).length,1,"one top-level SEO heading");
  assert.equal((page.match(/class="quant-tab-panel/g)||[]).length,3,"all three original calculator tabs retained");
  for(const id of ["bqStudioRoad","quantHistory","quantAddB","quantAddP","quantAddT","quantApply","quantUndo","quantClear","quantManualCard","quantPreviewCard"]){
    assert(page.includes('id="'+id+'"'),"functional original controls preserved: "+id);
  }
  assert(page.includes('href="bq-studio.css?v=20261122-studio-v1"'),"new self-hosted stylesheet");
  assert(page.includes('src="bq-studio.js?v=20261122-studio-v1"'),"new local read-only visualization");
  assert(page.includes('data-bq-studio-target="quantManualCard"')&&page.includes('data-bq-studio-target="quantPreviewCard"'),"user-selectable two-mode strip");
  assert(page.includes("本頁不會自動讀取外部牌桌")||page.includes("影片僅呈現過去錄製的監控畫面"),"public UI accurately discloses limitations on live table connections");
  assert(!page.includes('id="dgDirectTrialLink"')&&!page.includes("開啟外部免費試玩"),"removed external demo stays removed");
  assert(!page.includes("5,260,000")&&!page.includes("61%⭡")&&!page.includes("勝率 63%"),"no competitor's unverifiable data copied");
}
assert(home.includes("莊家剋星｜")&&home.includes("AI 監控與預測分析"),"莊家剋星 original video-first product copy");
assert(home.includes('id="bqYouTubeVideo"')&&home.includes("www.youtube-nocookie.com/embed/ns76j16hLTM")&&!home.includes('id="quantHeroFill"'),"hero embeds existing YouTube video rather than a fabricated profit chart or local MP4");
assert(home.includes("前往 YouTube 觀看")&&home.includes("不代表即時讀牌"),"YouTube fallback and non-live limitations are stated");
assert.equal((home.match(/id="trialStartBtn"/g)||[]).length,1,"real BBB trial remains single");
assert(home.includes('href="partner-entry.html"')&&home.includes('id="trialStartMessage"'),"partner and trial status remain");
assert(!tools.includes('id="trialStartBtn"'),"independent tools page links to real trial instead of duplicating auth buttons");
assert(tools.includes("尚未接入")&&tools.includes("目前此網站未連接遊戲商即時資料"),"auto scanning never claimed as available");
assert(css.includes("grid-template-columns:repeat(10,minmax(0,1fr))"),"readable responsive bead road");
assert(css.includes("@media(max-width:760px)")&&css.includes("prefers-reduced-motion:reduce"),"mobile and reduced-motion styles");
assert(!/fetch\s*\(|XMLHttpRequest|WebSocket|innerHTML/.test(code),"preview code is local, read-only and safe");

// Browser-independent interaction test: the road must mirror the visitor's
// actual input rather than pre-seeded historical or model prediction values.
const listeners=new Map();
function mockElement(id){
  return {
    id,value:"",attrs:{},children:[],className:"",dataset:{},
    classList:{toggle(name,on){this[name]=on;}},
    addEventListener(event,cb){listeners.set(id+":"+event,cb);},
    setAttribute(name,value){this.attrs[name]=value;},
    replaceChildren(...els){this.children=els.flatMap(v=>v.isFragment?v.children:[v]);},
    scrollIntoView(){this.scrolled=true;}
  };
}
const ids=["quantHistory","bqStudioRoad","quantAddB","quantAddP","quantAddT","quantApply","quantUndo","quantClear","quantManualCard","quantPreviewCard"];
const nodes=Object.fromEntries(ids.map(id=>[id,mockElement(id)]));
const buttons=[mockElement("manualMode"),mockElement("trialMode")];
buttons[0].dataset.bqStudioTarget="quantManualCard";
buttons[1].dataset.bqStudioTarget="quantPreviewCard";
const document={
  getElementById(id){return nodes[id]||null;},
  querySelectorAll(sel){return sel==="[data-bq-studio-target]"?buttons:[];},
  createElement(tag){return {...mockElement("bead"),tag,textContent:""};},
  createDocumentFragment(){return {isFragment:true,children:[],append(item){this.children.push(item);}};}
};
const window={BBBQuantMath:math,matchMedia(){return {matches:true};}};
vm.runInNewContext(code,{window,document,queueMicrotask}, {filename:"bq-studio.js"});
assert(nodes.bqStudioRoad.children.length===1,"no fake initial road beads");
assert.equal(nodes.bqStudioRoad.children[0].textContent,"輸入莊／閒／和後顯示實際紀錄");
nodes.quantHistory.value="B P T B";
listeners.get("quantHistory:input")();
assert.deepEqual(nodes.bqStudioRoad.children.map(x=>x.textContent),["莊","閒","和","莊"]);
assert(nodes.bqStudioRoad.attrs["aria-label"].includes("最近 4 局"),"a11y summary uses actual count");
nodes.quantHistory.value="B X";
listeners.get("quantHistory:input")();
assert.equal(nodes.bqStudioRoad.children[0].textContent,"輸入格式有誤，請使用莊／閒／和","invalid inputs never generate invented data");
listeners.get("trialMode:click")();
assert.equal(buttons[1].attrs["aria-pressed"],"true","accessible trial mode selection");
assert(nodes.quantPreviewCard.scrolled,"trial mode takes visitor to actual trial card");
assert.equal(nodes.quantPreviewCard.classList["bq-studio-highlight"],true,"selected trial card visibly highlighted");
console.log(JSON.stringify({ok:true,checks:["product_copy","real_30min_trial","no_external_free_demo","three_static_tabs","no_fake_live_signal","dynamic_actual_road","invalid_input_safe","two_mode_accessible","responsive_visuals","no_remote_calls"]}));

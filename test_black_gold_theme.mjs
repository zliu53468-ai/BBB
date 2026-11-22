/* CSS-only theme tests: visual design and copy without modifying forecast state. */
import assert from "node:assert/strict";
import fs from "node:fs";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const pages=["index.html","trial-ended.html","baccarat-road-guide.html","baccarat-ai-prediction.html","baccarat-ai-winrate.html","baccarat-kelly-guide.html","youtube-video.html"];
const css=read("black-gold-theme.css");
for(const page of pages){
 const content=read(page);
 assert(content.includes('black-gold-theme.css?v=20261116-blackgold-v1'),page+": theme loaded last");
 assert(content.includes("bbb-black-gold"),page+": black gold body class");
 assert(content.includes('name="theme-color" content="#0b0b09"'),page+": browser theme color");
 assert(!content.includes('src="assets/brand-ai-baccarat-hero.webp"'),page+": no portrait visual");
 assert(!content.includes('src="assets/hero-baccarat-slayer.webp"'),page+": no portrait foreground");
 if(page!=="youtube-video.html"){
  assert(!content.includes('property="og:image" content="https://bgsv1.xyz/assets/brand-ai-baccarat-hero.webp"'),page+": no old portrait share preview");
 }
}
const home=read("index.html"),exit=read("trial-ended.html");
assert(home.includes("開啟外部免費試玩"),"brand-neutral trial label");
assert(home.includes("想領取程式開通碼，請聯繫下方管理員"),"returning user exact heading");
assert(!home.includes("想領取程式開通碼，繼續使用 BBB 嗎？"),"outdated heading gone");
assert(home.includes('id="dgDirectTrialLink"')&&home.includes('data-bbb-link="dgTrialUrl"'),"DG independent route intact");
assert(home.includes('id="trialStartBtn"')&&home.includes('aria-label="立即試用莊家剋星預測程式｜免費試用 30 分鐘"'),"BBB trial button readable to assistive technology");
assert(home.includes("bbb-neon-cta-start")&&home.includes("bbb-neon-cta-dg"),"two independent neon cards");
assert(css.includes("grid-template-columns:repeat(2,minmax(0,1fr))"),"desktop two-card layout");
assert(css.includes("grid-template-columns:1fr"),"phone stacked layout");
assert(css.includes("--cta-edge:#4bd5a1")&&css.includes("--cta-edge:#449bff"),"reference-inspired green and blue rims");
assert(css.includes("#predictionApp .card"),"prediction panel colors changed");
assert(css.includes("#predictionApp .direction-orb"),"direction orb colors changed");
assert(css.includes(".bbb-gold-prediction-emblem"),"new person-free prediction artwork");
assert(css.includes("body.bbb-black-gold #licenseGate>.bbb-gold-landscape"),"black-gold animated homepage");
assert(css.includes("body.bbb-black-gold.trial-ended-body>.bbb-gold-landscape"),"same animated exit page");
assert(css.includes("@keyframes bbb-gold-light-sweep"),"gold sweep animation");
assert(css.includes("prefers-reduced-motion:reduce"),"motion reduction");
assert(home.includes('class="bbb-site-tabs"'),"three-tab site remains");
assert(home.includes('id="bbbFeedbackLightbox"'),"zoom dialog retained");
for(const id of ["btnB","btnP","btnT","btnBack","btnStart","bptBigRoad","bbbTrialAccuracyHits","bbbTrialAccuracyValid","bbbTrialCountdown","bbbBankrollDialog"])
 assert(home.includes('id="'+id+'"'),"preserved model/prediction interface "+id);
assert(home.includes('src="soft_shoe_decision_runtime.js')===false,"prediction runtime is loaded by existing auth/trial scripts, not replaced");
assert(exit.includes('data-bbb-link="casinoRegisterUrl"')&&exit.includes('data-bbb-link="lineContactUrl"'),"conversion buttons intact");
assert(exit.includes('id="casinoAccountDialog"'),"transfer modal intact");
console.log(JSON.stringify({ok:true,checks:["seven_page_black_gold","person_free_main_hero","DG_exact_cta","returning_copy","desktop_double_neon","mobile_stack","prediction_panel_gold","responsive_motion","unchanged_routes"]}));

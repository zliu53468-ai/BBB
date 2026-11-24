import assert from "node:assert/strict";
import fs from "node:fs";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const home=read("index.html"),tool=read("quant-tools.html");
const video=fs.readFileSync(new URL("./assets/banker-slayer-monitor-demo.mp4",import.meta.url));
assert(video.length>=500000&&video.length<=800000,"original uploaded monitor video retained at readable quality");
assert(video.subarray(0,64).includes(Buffer.from("ftyp")),"self-hosted source is a valid MP4 container header");
assert.equal((home.match(/id="bqMonitorVideo"/g)||[]).length,1,"one real monitor video");
assert(home.includes('<video id="bqMonitorVideo" autoplay muted loop playsinline controls'),"responsive, muted mobile-friendly playback with native pause");
assert(home.includes('<source src="assets/banker-slayer-monitor-demo.mp4" type="video/mp4">'),"the exact committed asset is referenced");
assert(home.includes("監控介面的錄製片段")&&home.includes("不是即時訊號"),"video provenance clearly disclosed");
assert(home.includes("影片不代表本站正在即時連接遊戲商"),"no fabricated live connection");
assert(home.includes("AI 監控與預測分析"),"hero brand-first messaging");
assert(home.includes('id="quantAnalyzer"')&&home.includes('id="monitorVideo"'));
assert(home.indexOf('id="monitorVideo"')<home.indexOf('id="quantAnalyzer"'),"monitor video directly precedes trial section");
assert(home.indexOf('id="quantAnalyzer"')<home.indexOf('id="quantTopics"'),"trial before topic cluster");
assert(home.includes('<details class="bq-home-advanced"'),"manual road retained as optional disclosure");
assert(home.indexOf('id="bqAdvancedManual"')<home.indexOf('id="quantManualCard"'),"advanced manual card is not in initial visual journey");
assert(home.indexOf('id="quantPreviewCard"')<home.indexOf('id="quantManualCard"'),"real trial before manual research");
assert.equal((home.match(/id="trialStartBtn"/g)||[]).length,1,"existing protected trial button unique");
assert.equal((home.match(/id="dgDirectTrialLink"/g)||[]).length,0,"previously removed public demo CTA remains absent");
assert(home.includes("partner-entry.html"),"existing partner portal preserved");
for(const id of ["quantHistory","quantAddB","quantAddP","quantAddT","quantEvB","quantSimChart","licenseForm","predictionApp","btnStart","btnB","btnP","btnT"]){
 assert(home.includes('id="'+id+'"'),"existing app, fallback and calculator controls still in DOM: "+id);
}
assert.equal((home.match(/class="quant-tab-panel/g)||[]).length,3,"all original research tools remain source rendered");
assert(tool.includes('id="quantHistory"')&&tool.includes('id="bqStudioRoad"'),"standalone analysis page retains manual road tools");
for(const p of ["index.html","quant-tools.html","research-hub.html","model-methodology.html","risk-center.html","partner-entry.html","research-martingale-100k.html","research-shoe-deviation.html","research-scam-safety-2026.html"]){
 const source=read(p);
 assert(source.includes('<span class="quant-site-brand-copy">莊家剋星'),"unified brand in "+p);
 assert(!source.includes("BaccaratQuant"),"no prior public brand in "+p);
}
const videoCSS=read("bq-video-first.css");
assert(videoCSS.includes(".bq-monitor-hero")&&videoCSS.includes("@media(max-width:760px)"),"video-first responsive style exists");
assert(videoCSS.includes(".bq-home-advanced")&&videoCSS.includes(".bq-home-trial-layout"),"optional manual and primary trial styling");
console.log(JSON.stringify({ok:true,checks:["self_hosted_original_mp4","video_position","real_trial_preserved","manual_optional","standalone_tools","model_controls_intact","unified_brand","no_live_feed_misrepresentation","responsive_mobile"]}));

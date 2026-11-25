import assert from "node:assert/strict";
import fs from "node:fs";
const read=path=>fs.readFileSync(new URL("./"+path,import.meta.url),"utf8");
const home=read("index.html"),tool=read("quant-tools.html"),hub=read("research-hub.html");
const videoCSS=read("bq-video-first.css");

// Official YouTube watch URL is already configured elsewhere in the site.
// Playback is embedded; upload and YouTube live streaming require channel-side setup.
const watch="https://www.youtube.com/watch?v=ns76j16hLTM";
const embed="https://www.youtube-nocookie.com/embed/ns76j16hLTM";
assert.equal((home.match(/id="bqYouTubeVideo"/g)||[]).length,1,"one YouTube iframe in the hero");
assert(home.includes('src="'+embed+'?rel=0&amp;playsinline=1"'),"YouTube privacy-enhanced player embeds the established video");
assert(home.includes("allowfullscreen")&&home.includes('loading="lazy"'),"accessible lazy YouTube embed");
assert(home.includes(watch),'open on YouTube fallback link');
assert(home.includes("年齡、地區或影片設定無法嵌入"),"clear explanation of embedded playback limitations");
assert(!/<video\b/i.test(home)&&!/<video\b/i.test(tool),"no local inline MP4 player on home or tools");
assert(!home.includes('src="assets/banker-slayer-monitor-demo.mp4"'),"local recorded clip removed from UI");
assert(!tool.includes('bqMonitorToolsVideo')&&!tool.includes('class="bq-tools-video"'),"cramped monitoring preview removed from trial card");
assert(tool.includes(watch)&&tool.includes("前往首頁開始 30 分鐘試用"),"tools provide YouTube link and actual trial entry");
assert(videoCSS.includes(".bq-youtube-player")&&videoCSS.includes("aspect-ratio:16/9"),"wide responsive 16:9 video");
assert(videoCSS.includes(".bq-youtube-player iframe")&&videoCSS.includes("@media(max-width:760px)"),"iframe fills frame on desktop and mobile");
assert(home.includes('bq-video-first.css?v=20261125-youtube-v2')&&tool.includes('bq-video-first.css?v=20261125-youtube-v2'),"updated stylesheet served without old cache");
assert(home.includes("非即時讀牌")||home.includes("不代表即時讀牌"),"video does not misrepresent live data");
assert(home.includes("AI 監控與預測分析"),"hero brand copy maintained");
assert(home.includes('id="quantAnalyzer"')&&home.includes('id="monitorVideo"'));
assert(home.indexOf('id="monitorVideo"')<home.indexOf('id="quantAnalyzer"'),"YouTube hero precedes trial");
assert(home.indexOf('id="quantAnalyzer"')<home.indexOf('id="quantTopics"'),"trial before topic cluster");
assert(home.includes('<details class="bq-home-advanced"'),"manual road stays optional");
assert(home.indexOf('id="bqAdvancedManual"')<home.indexOf('id="quantManualCard"'),"manual input is behind advanced disclosure");
assert(home.indexOf('id="quantPreviewCard"')<home.indexOf('id="quantManualCard"'),"real trial precedes advanced manual input");
assert.equal((home.match(/id="trialStartBtn"/g)||[]).length,1,"actual authorized trial start button unique");
assert(!home.includes('id="dgDirectTrialLink"'),"previously removed public free-demo CTA stays absent");
assert(home.includes("partner-entry.html"),"partner service route remains");
for(const id of ["quantHistory","quantAddB","quantAddP","quantAddT","quantEvB","quantSimChart","licenseForm","predictionApp","btnStart","btnB","btnP","btnT"]){
  assert(home.includes('id="'+id+'"'),"core model / research entry stays in DOM: "+id);
}
assert.equal((home.match(/class="quant-tab-panel/g)||[]).length,3,"three original research tabs remain in HTML");
assert(tool.includes('id="quantHistory"')&&tool.includes('id="bqStudioRoad"'),"independent tools retain optional manual research");
// No fake benchmark rows: keep the methodology anchor and short explanation, not blank comparisons.
for(const html of [home,hub]){
 assert(html.includes('id="quantBacktest"'),"transparent methodology anchor remains");
 assert(!html.includes("待實測")&&!html.includes("最大回撤 Max Drawdown"),"unfinished performance comparison is not shown");
 assert(html.includes('href="model-methodology.html"'),"reader can check published model limitations");
}
for(const path of ["index.html","quant-tools.html","research-hub.html","model-methodology.html","risk-center.html","partner-entry.html","research-martingale-100k.html","research-shoe-deviation.html","research-scam-safety-2026.html"]){
 const s=read(path);
 assert(s.includes('<span class="quant-site-brand-copy">莊家剋星'),"unified public brand: "+path);
 assert(!s.includes("BaccaratQuant"),"no old brand: "+path);
}
console.log(JSON.stringify({ok:true,checks:["privacy_youtube_embed","responsive_16_9","no_narrow_local_video","age_restricted_fallback_link","trial_untouched","honest_evidence","no_unverified_drawdown_table","public_brand_consistent"]}));

// Four screenshot-based BBB UI refinements with no model/authorization changes.
import fs from "node:fs";
import assert from "node:assert/strict";
const read=name=>fs.readFileSync(new URL("./"+name,import.meta.url),"utf8");
const html=read("index.html"),css=read("trial-styles.css");
const engine=read("soft_shoe_decision_runtime.js"),panel=read("soft_shoe_panel.js");
const trial=read("trial-client.js"),auth=read("auth-client.js");

assert(html.includes('class="license-logo dynamic-brand-mark"'),"existing brand logo receives dynamic wrapper");
assert(html.includes('src="assets/logo-baccarat-slayer.webp"'),"keep original avatar image file");
assert(html.includes('class="homepage-impact-line"'),"strong branded headline subtext");
assert(html.includes('class="homepage-gradient-title"'),"original SEO brand identity kept");
for(const key of ["bbb-avatar-halo","bbb-avatar-float","bbb-title-shine"]){
  assert(css.includes("@keyframes "+key+"{"),"dynamic CSS animation: "+key);
}
assert(css.includes("prefers-reduced-motion:reduce"),"motion accessibility");
assert(/<p id="trialStartMessage"[^>]*><\/p>/.test(html),"screenshot 1 verbose default instruction removed");
assert(css.includes("#trialStartMessage:empty{display:none!important}"),"blank instruction takes zero space");
assert(trial.includes('function showError(message)'),"trial errors still surfaced through existing status element");
assert(html.includes('class="seo-spotlight"'),"screenshot 2 concise spotlight");
assert((html.match(/class="seo-feature-card"/g)||[]).length===3,"three visual SEO summary cards");
assert(html.includes('<details class="seo-compact-details">'),"long form FAQ is collapsed but crawlable");
assert(html.includes('href="baccarat-road-guide.html"'),"original SEO internal link preserved");
assert(html.includes("百家樂大路")&&html.includes("百家樂路單"),"important real-content search phrases preserved");
assert(!html.includes('<h2>百家樂AI分析器：莊閒和、百家路單與百家打法研究</h2>'),"old verbose screenshot 2 block replaced");
assert(html.includes('<div id="message" aria-live="off" hidden></div>'),"screenshot 3 diagnostic UI visually removed");
assert(css.includes("#predictionApp #message{display:none!important}"),"hidden diagnostic never reappears visually");
assert(css.includes("#appViewportContent::-webkit-scrollbar{display:none!important"),"screenshot 4 WebKit scrollbar hidden");
assert(css.includes("scrollbar-width:none"),"Firefox scrollbar hidden");
assert(css.includes("overflow-y:auto!important"),"mobile wheel and touch scrolling preserved");
assert(!css.includes("#appViewportContent{display:none"),"app is never hidden");
for(const key of ["trialStartBtn","dgDirectTrialLink","licenseForm","bbbTrialAccuracyPanel","predictionApp"]){
  assert(html.includes('id="'+key+'"'),"existing interactive element preserved: "+key);
}
assert(panel.includes('function recalculate()')||panel.includes("async function recalculate()"),"original prediction panel algorithm retained");
assert(engine.includes("BBB_SOFT_SHOE_BPT_EV_V5_1"),"original prediction engine version retained");
assert(auth.includes("license"),"existing member auth retained");
assert(html.includes('trial-styles.css?v=20261116-ui-polish-v7'),"cache version bumped");
console.log(JSON.stringify({ok:true,checks:["animated_original_avatar","impactful_neon_heading","reduced_motion","remove_default_instruction","seo_cards","expandable_FAQ","diagnostics_hidden","scrollbar_invisible_but_scrollable","unchanged_ui_hooks"]}));

import assert from "node:assert/strict";
import fs from "node:fs";
const read=(p)=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const html=read("partner-entry.html");
const css=read("partner-entry.css");
const sitemap=read("sitemap.xml");
const title=html.match(/<title>([^<]+)<\/title>/)?.[1]||"";
const h1=html.match(/<h1[^>]*>([^<]+)<\/h1>/)?.[1]||"";
assert(title.includes("百家樂 AI 試用")&&title.includes("官方 LINE"),"search-intent title");
assert(h1.includes("百家樂 AI 試用"),"clear H1");
assert.equal((html.match(/<h1\b/g)||[]).length,1,"one H1");
assert(html.includes('rel="canonical" href="https://bgsv1.xyz/partner-entry.html"'),"canonical unchanged");
assert(html.includes('name="description" content="莊家剋星百家樂 AI 程式試用後'),"description matches body");
assert(html.includes('id="partnerGuide"')&&html.includes('id="bqPartnerGuideHeading"'),"semantic in-source guide");
assert.equal((html.match(/class="bq-portal-help-item(?: bq-portal-help-wide)?"/g)||[]).length,5,"five distinct informative articles");
assert.equal((html.match(/<h3>/g)||[]).length,5,"five searchable question headings");
for(const phrase of [
  "百家樂 AI 預測程式要怎麼開始 30 分鐘試用",
  "如何聯絡官方 LINE 領取開通資訊",
  "娛樂城合作平台註冊前",
  "已經有娛樂城會員帳號",
  "監控影片是即時讀牌嗎",
  "不保證平台合法性或收益",
  "不是已接入外部牌桌的即時串流",
]){
  assert(html.includes(phrase),"natural answer and keyword coverage: "+phrase);
}
for(const href of ["./#quantAnalyzer","research-scam-safety-2026.html","./#monitorVideo","model-methodology.html"]){
  assert(html.includes('href="'+href+'"'),"helpful internal crosslink "+href);
}
assert(html.includes('data-bbb-link="casinoRegisterUrl"')&&html.includes('data-bbb-link="lineContactUrl"'),"original outbound links not replaced");
for(const id of ["casinoAccountDialog","casinoAccountClose","casinoAccountDismiss","casinoNextHint"]){
  assert(html.includes('id="'+id+'"'),"existing companion action kept "+id);
}
assert(html.includes('src="site-links.js"')&&html.includes('src="outbound-links.js?v=20261115-return-rotation-v3"'),"existing link rotation preserved");
assert(html.indexOf('class="bq-portal-grid"')<html.indexOf('id="partnerGuide"'),"conversion choices stay first, guide follows");
assert(html.includes('href="partner-entry.css?v=20261125-longtail-v2"'),"mobile browser loads updated CSS");
assert(css.includes(".bq-portal-help-grid")&&css.includes("@media(max-width:760px)")&&css.includes("grid-template-columns:1fr"),"responsive long-tail articles");
assert(sitemap.includes("<loc>https://bgsv1.xyz/partner-entry.html</loc><lastmod>2026-11-25</lastmod>"),"sitemap fresh");
const schemas=[...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map(x=>JSON.parse(x[1]));
assert.equal(schemas.length,1,"webpage metadata parses");
assert(schemas[0].name.includes("官方 LINE"),"structured page metadata aligned");
assert(!html.includes('meta name="keywords"'),"no keyword-stuffing meta tags");
console.log(JSON.stringify({ok:true,checks:["title_and_body_longtails","five_source_rendered_answers","one_H1","natural_internal_links","original_partner_rotator_kept","responsive_layout","canonical_and_sitemap"]}));

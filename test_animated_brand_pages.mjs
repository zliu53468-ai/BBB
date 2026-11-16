// Only the two promotional pages, their branded asset, and CTA labels are changed.
import fs from "node:fs";
import assert from "node:assert/strict";
const read=path=>fs.readFileSync(new URL("./"+path,import.meta.url),"utf8");
const main=read("index.html"),ended=read("trial-ended.html"),css=read("trial-styles.css");
const binary=fs.readFileSync(new URL("./assets/brand-ai-baccarat-hero.webp",import.meta.url));
const image="assets/brand-ai-baccarat-hero.webp";
assert(binary.length>5000,"uploaded brand artwork must exist, not an empty placeholder");
assert.equal(binary.subarray(0,4).toString("ascii"),"RIFF","brand image RIFF header");
assert.equal(binary.subarray(8,12).toString("ascii"),"WEBP","brand image WebP signature");
assert(main.includes('<img class="bbb-animated-brand-image" src="'+image+'"'),
  "homepage uses the exact user-provided branded image");
assert(ended.includes('<img class="bbb-animated-brand-image" src="'+image+'"'),
  "trial-ended casino + LINE page uses matching image");
assert(main.includes('id="trialStartBtn">立即試用莊家剋星預測程式｜免費試用 30 分鐘</button>'),
  "homepage CTA exactly updated");
assert(main.includes('class="bbb-bankroll-confirm">立即試用測試</button>'),
  "capital confirmation CTA updated");
assert(main.includes('id="bbbBankrollDialog"'),"original bankroll modal preserved");
assert(main.includes('id="trialStartMessage"'),"existing trial feedback preserved");
assert(ended.includes('data-bbb-link="casinoRegisterUrl"')&&ended.includes('data-bbb-link="lineContactUrl"'),
  "existing casino and LINE destinations preserved");
assert(ended.includes('id="casinoAccountDialog"'),"existing-account prompt preserved");
assert(css.includes("@keyframes bbb-brand-kenburns"),"slow dynamic zoom");
assert(css.includes("@keyframes bbb-brand-light-sweep"),"purple sweeping light");
assert(css.includes("@keyframes bbb-brand-frame-pulse"),"neon frame motion");
assert(css.includes("prefers-reduced-motion:reduce"),"accessible reduced-motion fallback");
assert(main.includes('trial-styles.css?v=20261116-brandmotion-v8')&&
  ended.includes('trial-styles.css?v=20261116-brandmotion-v8'),"both pages cache-busted");
assert(main.includes('id="bbbTrialAccuracyPanel"'),"60 percent accuracy gate visible in trial");
console.log(JSON.stringify({ok:true,checks:["brand_image_binary","homepage_brand_motion","trial_end_brand_motion","cta_home","cta_bankroll","casino_and_line_intact","trial_untouched","reduced_motion"]}));

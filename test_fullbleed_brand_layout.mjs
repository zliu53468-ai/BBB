import fs from "node:fs";
import assert from "node:assert/strict";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const home=read("index.html");
const end=read("trial-ended.html");
const css=read("trial-styles.css");
const image="assets/brand-ai-baccarat-hero.webp";

assert(home.includes('id="licenseGate"'),"original licensed landing remains");
assert(end.includes('class="trial-ended-body"'),"trial end landing remains");
assert(home.includes('<section id="licenseGate" class="license-gate"'),"original landing ID unchanged");
assert(home.includes('<div class="bbb-fullbleed-scene" aria-hidden="true">'),"home has full screen scene");
assert(end.includes('<div class="bbb-fullbleed-scene" aria-hidden="true">'),"exit has full screen scene");
assert(home.indexOf('class="bbb-fullbleed-scene"')<home.indexOf('class="license-shell"'),
  "scene sits outside overlay content card, not inside a small image box");
assert(end.indexOf('class="bbb-fullbleed-scene"')<end.indexOf('class="trial-end-shell"'),
  "exit background covers page, not conversion card");
assert(home.includes('src="'+image+'"')&&end.includes('src="'+image+'"'),
  "original supplied brand artwork reused on both pages");
assert(!home.includes('class="bbb-animated-brand-art"'),"old square homepage image card removed");
assert(!end.includes('class="bbb-animated-brand-art bbb-end-brand-art"'),"old square trial end card removed");
assert(css.includes("#licenseGate>.bbb-fullbleed-scene"),"homepage fixed image background selectors");
assert(css.includes(".trial-ended-body>.bbb-fullbleed-scene"),"trial end fixed image background selectors");
assert(css.includes("position:fixed;inset:0;width:100%"),"scene covers viewport rather than 430px");
assert(css.includes("object-fit:cover"),"uploaded image covers whole viewport");
assert(css.includes("@keyframes bbb-fullbleed-drift"),"slow parallax effect");
assert(css.includes("@keyframes bbb-fullbleed-sweep"),"purple light sweep");
assert(css.includes("pointer-events:none"),"full-screen artwork cannot block buttons");
assert(css.includes("prefers-reduced-motion:reduce"),"motion respects accessibility");
assert(home.includes('id="trialStartBtn"'),"BBB trial CTA preserved");
assert(home.includes('id="dgDirectTrialLink"'),"DG independent CTA preserved");
assert(home.includes('id="bbbBankrollDialog"'),"capital modal preserved");
assert(home.includes('id="bbbTrialAccuracyPanel"'),"60% hit-rate trial display preserved");
assert(end.includes('data-bbb-link="casinoRegisterUrl"'),"casino redirect preserved");
assert(end.includes('data-bbb-link="lineContactUrl"'),"LINE redirect preserved");
assert(end.includes('id="casinoAccountDialog"'),"existing account prompt preserved");
console.log(JSON.stringify({ok:true,checks:["full_bleed_home","full_bleed_exit","same_original_artwork","no_square_card","interactive_controls_preserved","scroll_and_motion_accessible"]}));

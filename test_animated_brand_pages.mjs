// Existing brand artwork is kept as a repository asset but is no longer rendered
// as a foreground figure or the OpenGraph image. These are visual-only changes.
import assert from "node:assert/strict";
import fs from "node:fs";
const read=name=>fs.readFileSync(new URL("./"+name,import.meta.url),"utf8");
const home=read("index.html"),ended=read("trial-ended.html");
const old=fs.readFileSync(new URL("./assets/brand-ai-baccarat-hero.webp",import.meta.url));
const css=read("black-gold-theme.css"),base=read("trial-styles.css");
assert(old.length>5000&&old.subarray(0,4).toString("ascii")==="RIFF"&&
 old.subarray(8,12).toString("ascii")==="WEBP","old user-supplied art preserved as an archive");
assert(!home.includes('src="assets/brand-ai-baccarat-hero.webp"'),"homepage does not feature the person");
assert(!ended.includes('src="assets/brand-ai-baccarat-hero.webp"'),"trial-ended page does not feature the person");
assert(!home.includes('src="assets/hero-baccarat-slayer.webp"'),"prediction panel does not feature the person");
assert(home.includes('class="bbb-fullbleed-scene bbb-gold-landscape"'),"gold abstract landscape on homepage");
assert(ended.includes('class="bbb-fullbleed-scene bbb-gold-landscape"'),"gold abstract landscape on end page");
assert(home.includes('class="brand-hero bbb-gold-prediction-hero"'),"prediction panel has gold graphic");
assert(home.includes('class="bbb-neon-cta bbb-neon-cta-start"')||home.includes('class="bbb-neon-cta bbb-neon-cta-start"'),"new black neon start card");
assert(home.includes('bbb-neon-cta-dg'),"new black neon DG card");
assert(home.includes("開啟外部免費試玩"),"brand-neutral external trial label");
assert(home.includes('id="trialStartBtn"'),"old BBB start hook kept");
assert(home.includes('class="bbb-bankroll-confirm">立即試用測試</button>'),"capital confirmation unchanged");
assert(home.includes('id="bbbBankrollDialog"'),"capital modal unchanged");
assert(home.includes('id="trialStartMessage"'),"start message unchanged");
assert(ended.includes('data-bbb-link="casinoRegisterUrl"')&&ended.includes('data-bbb-link="lineContactUrl"'),"end page outbound actions unchanged");
assert(ended.includes('id="casinoAccountDialog"'),"old transfer prompt unchanged");
assert(css.includes("@keyframes bbb-gold-light-sweep"),"gold light sweep animation");
assert(css.includes("@keyframes bbb-gold-orbit"),"gold concentric orbit animation");
assert(css.includes("@keyframes bbb-gold-hero-float"),"prediction banner animation");
assert(css.includes("prefers-reduced-motion:reduce"),"reduced motion included");
assert(home.includes('black-gold-theme.css?v=20261116-blackgold-v1')&&
 ended.includes('black-gold-theme.css?v=20261116-blackgold-v1'),"new theme loaded both pages");
assert(base.includes("prefers-reduced-motion:reduce"),"old reduced motion fallback remains");
console.log(JSON.stringify({ok:true,checks:["keeps_archived_artwork","no_visible_people","black_gold_home_exit_panel","DG_exact_label","neon_CTAs","preserves_auth_and_conversion","reduced_motion"]}));

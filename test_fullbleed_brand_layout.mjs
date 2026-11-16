import assert from "node:assert/strict";
import fs from "node:fs";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const home=read("index.html"),exit=read("trial-ended.html"),css=read("black-gold-theme.css"),old=read("trial-styles.css");
assert(home.includes('id="licenseGate"'),"original licensed entry remains");
assert(exit.includes('class="trial-ended-body bbb-black-gold"'),"full-bleed exit remains");
assert(home.includes('class="bbb-fullbleed-scene bbb-gold-landscape"'),"homepage uses abstract black-gold full page scene");
assert(exit.includes('class="bbb-fullbleed-scene bbb-gold-landscape"'),"exit page uses same abstract scene");
assert(home.indexOf('class="bbb-fullbleed-scene bbb-gold-landscape"')<
 home.indexOf('class="license-shell"'),"full-bleed home scene outside form card");
assert(exit.indexOf('class="bbb-fullbleed-scene bbb-gold-landscape"')<
 exit.indexOf('class="trial-end-shell"'),"full-bleed exit scene outside CTA card");
assert(!home.includes('class="bbb-animated-brand-art"')&&!exit.includes('class="bbb-animated-brand-art"'),
 "no small boxed hero image remains");
assert(!home.includes('src="assets/brand-ai-baccarat-hero.webp"')&&
 !exit.includes('src="assets/brand-ai-baccarat-hero.webp"'),"no illustrated hooded person is visible");
assert(css.includes("position:fixed;inset:0;width:100vw;height:100dvh"),"viewport fixed cover");
assert(css.includes("#licenseGate>.bbb-gold-landscape::before"),"gold particles override previous violet");
assert(css.includes("bbb-gold-light-sweep"),"gold sweep is animated");
assert(css.includes("pointer-events:none"),"background does not capture taps");
assert(css.includes("prefers-reduced-motion:reduce"),"motion accessibility");
assert(old.includes(".bbb-fullbleed-scene::before"),"base effects retained only as visually overridden CSS");
for(const id of ["trialStartBtn","dgDirectTrialLink","bbbBankrollDialog","bbbTrialAccuracyPanel"])
 assert(home.includes('id="'+id+'"'),"old trial control "+id);
for(const action of ["casinoRegisterUrl","lineContactUrl"])
 assert(exit.includes('data-bbb-link="'+action+'"'),"old external link "+action);
assert(exit.includes('id="casinoAccountDialog"'),"account prompt remains");
console.log(JSON.stringify({ok:true,checks:["full_viewport_gold_scene","no_portrait","responsive","keeps_controls","accessible_motion"]}));

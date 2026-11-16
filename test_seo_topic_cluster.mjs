// Original public SEO/AEO/GEO topic-cluster regression; no network or ranking claims.
import assert from "node:assert/strict";
import fs from "node:fs";
const read=p=>fs.readFileSync(new URL("./"+p,import.meta.url),"utf8");
const pages=[
 ["index.html","https://bgsv1.xyz/"],
 ["baccarat-ai-prediction.html","https://bgsv1.xyz/baccarat-ai-prediction.html"],
 ["baccarat-ai-winrate.html","https://bgsv1.xyz/baccarat-ai-winrate.html"],
 ["baccarat-road-guide.html","https://bgsv1.xyz/baccarat-road-guide.html"],
 ["baccarat-kelly-guide.html","https://bgsv1.xyz/baccarat-kelly-guide.html"],
 ["youtube-video.html","https://bgsv1.xyz/youtube-video.html"]
];
const titles=new Set(),descriptions=new Set(),sitemap=read("sitemap.xml"),robots=read("robots.txt");
const intro=read("index.html");
assert(intro.includes('property="og:site_name" content="莊家剋星"'),"social brand");
assert(intro.includes('property="og:image" content="https://bgsv1.xyz/assets/logo-baccarat-slayer.webp"'),"social share uses non-person brand mark");
assert(intro.includes('"@type":"WebSite"'),"Website brand schema");
assert(intro.includes('"name":"莊家剋星"'),"structured name");
assert(!intro.includes('<meta name="keywords"'),"no ignored meta keywords");
assert(intro.includes('class="member-feedback-showcase" data-nosnippet'),"screenshots not in snippets");
assert(intro.includes('class="seo-compact-copy" data-nosnippet'),"technical text excluded");
assert(intro.includes('id="message" aria-live="off" hidden data-nosnippet'),"diagnostic excluded");
assert(!intro.includes('id="predictionApp" hidden data-nosnippet'),"valid nosnippet tags only");
assert(intro.includes('id="trialStartBtn"')&&intro.includes('id="dgDirectTrialLink"'),"trial and DG links untouched");
assert(intro.includes('id="bbbTrialAccuracyPanel"'),"accuracy gate untouched");
for(const [path,url] of pages){
 const html=read(path);
 const title=html.match(/<title>([^<]+)<\/title>/)?.[1];
 const desc=html.match(/<meta name="description" content="([^"]+)"/)?.[1];
 assert(title&&title.length>=17&&title.length<=100,path+": meaningful title");
 assert(desc&&desc.length>=38&&desc.length<=200,path+": unique description");
 assert(!titles.has(title),path+": unique title");titles.add(title);
 assert(!descriptions.has(desc),path+": unique description");descriptions.add(desc);
 assert(html.includes('rel="canonical" href="'+url+'"'),path+": canonical");
 assert(html.includes('name="robots" content="index,follow'),path+": index allowed");
 assert(sitemap.includes("<loc>"+url+"</loc>"),path+": sitemap");
 assert(!/<meta[^>]*name="keywords"/.test(html),path+": no keyword spam tag");
 assert(!/<title>[^<]*(PF-EOR|EV-Deplete|V4|V5)/i.test(html),path+": public title no model version");
 if(path!=="index.html"){
   assert((html.match(/<h1\b/g)||[]).length===1,path+": single H1");
   assert(html.includes('href="./"'),path+": homepage link");
 }
}
const slugs=["baccarat-ai-prediction.html","baccarat-ai-winrate.html","baccarat-kelly-guide.html"];
for(const slug of slugs){
 const html=read(slug);
 assert(html.includes('"@type":"BreadcrumbList"'),slug+": BreadcrumbList");
 assert(html.includes('"@type":"WebPage"'),slug+": WebPage");
 assert(html.includes('class="seo-article-related"'),slug+": related links");
 assert(html.includes('class="seo-article-lead"'),slug+": visible intro");
 assert(html.length>3000,slug+": informative original article");
 assert(intro.includes('href="'+slug+'"'),slug+": linked by homepage");
}
assert(!sitemap.includes("trial-ended.html"),"trial conversion excluded");
assert(robots.includes("Sitemap: https://bgsv1.xyz/sitemap.xml"),"robots sitemap");
assert(intro.includes('trial-styles.css?v=20261116-fullbleed-v9'),"visuals untouched");
assert(read("seo-content.css").includes("#licenseGate .seo-topic-cluster"),"nav scoped to landing");
console.log(JSON.stringify({ok:true,checks:["six_canonical_pages","unique_titles_and_descriptions","topic_links","website_schema","breadcrumb_schema","clean_snippets","sitemap_and_robots"]}));

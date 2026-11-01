// Test-only regression fixture pinned to the 2026-10-31 production configuration.
const fs=require("fs"),path=require("path"),read=fs.readFileSync;
const originalProfile=JSON.parse(String(read.call(fs,path.join(__dirname,"validation/cardcount_profile_before_anti_streak_v1.json"),"utf8")));
fs.readFileSync=function(file,...args){
 const raw=read.call(this,file,...args);
 if(typeof file!=="string"||path.basename(file)!=="cardcount_profile.json")return raw;
 const p=JSON.parse(JSON.stringify(originalProfile));
 if(p.bayes_markov_final_v1)p.bayes_markov_final_v1.enabled=false;
 if(p.road_primary_v1)p.road_primary_v1.enabled=false;
 if(p.hitrate_max_v1?.markov)p.hitrate_max_v1.markov.enabled=true;
 const output=JSON.stringify(p);
 return Buffer.isBuffer(raw)?Buffer.from(output):output;
};

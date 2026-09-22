#!/usr/bin/env node
"use strict";

/*
Build synthetic residual-training rows with the EXISTING production core.
This script does not reimplement V23; it loads app256forward.js +
app256continuation.js and calls hazardChoose(history) directly.
*/

const fs = require("fs");

global.window = global;
const store = new Map();
global.localStorage = {
  getItem: k => store.has(k) ? store.get(k) : null,
  setItem: (k,v) => store.set(k,String(v)),
  removeItem: k => store.delete(k),
};
global.document = {
  getElementById: () => null,
};

require("./app256forward.js");
require("./app256continuation.js");

const CORE = global.__BGS256_CONTINUATION_TEST__;
if (!CORE || typeof CORE.hazardChoose !== "function") {
  throw new Error("V23 core failed to load");
}

const clip = (v,lo=0,hi=1) => Math.max(lo,Math.min(hi,Number.isFinite(+v)?+v:lo));
const bp = seq => seq.filter(x => x==="B" || x==="P");

function transitions(seq){
  const a=bp(seq), out=[];
  for(let i=1;i<a.length;i++) out.push(a[i]===a[i-1]?"S":"X");
  return out;
}
function sxMarkovPSame(seq,window=24,prior=1){
  const t=transitions(seq);
  if(!t.length) return .5;
  const current=t.at(-1), start=Math.max(0,t.length-1-Math.max(2,window));
  let same=0,sw=0;
  for(let i=start;i<t.length-1;i++){
    if(t[i]!==current) continue;
    if(t[i+1]==="S") same++; else if(t[i+1]==="X") sw++;
  }
  return clip((same+prior)/(same+sw+2*prior));
}
function stage(seq){
  const a=bp(seq); if(!a.length) return 0;
  const side=a.at(-1); let n=1;
  for(let i=a.length-2;i>=0 && a[i]===side;i--) n++;
  return n;
}
function depth(seq){
  const t=transitions(seq); if(!t.length) return 0;
  const token=t.at(-1); let n=1;
  for(let i=t.length-2;i>=0 && t[i]===token;i--) n++;
  return n;
}
function original7(corePB,seq){
  const roundIndex=Math.max(1,Math.min(70,seq.length+1));
  const estimatedTotalHands=60;
  const remainingRatio=clip((estimatedTotalHands-(roundIndex-1))/estimatedTotalHands);
  return [
    corePB,
    roundIndex,
    estimatedTotalHands,
    remainingRatio,
    sxMarkovPSame(seq),
    stage(seq),
    depth(seq),
  ];
}

const inputPath=process.argv[2];
const outputPath=process.argv[3] || "synthetic_core_rows.json";
if(!inputPath) throw new Error("usage: node build_synthetic_residual_rows.js simulated_rows.json output.json");

const payload=JSON.parse(fs.readFileSync(inputPath,"utf8"));
const rows=Array.isArray(payload) ? payload : payload.rows;
if(!Array.isArray(rows)) throw new Error("input must contain rows");

const output=[];
for(let i=0;i<rows.length;i++){
  const row=rows[i] || {};
  const seq=String(row.history || "").split("").filter(x=>["B","P","T"].includes(x));
  if(!seq.length) continue; // V23 needs an existing road state.
  const actual=String(row.actual_outcome || "").toUpperCase();
  if(actual!=="B" && actual!=="P") continue; // residual target is directional only.
  const prediction=CORE.hazardChoose(seq);
  const corePB=clip(+prediction?.probabilities?.B || .5);
  const f=original7(corePB,seq);
  output.push({
    schema_version:2,
    shoe_id:String(row.shoe_id ?? ""),
    history:seq.join(""),
    history_fingerprint:seq.join(""),
    actual_outcome:actual,
    actual_b:actual==="B"?1:0,
    core_p_b:corePB,
    round_index:f[1],
    estimated_total_hands:f[2],
    remaining_ratio:f[3],
    sx_markov_p_same:f[4],
    stage:f[5],
    depth:f[6],
  });
}
fs.writeFileSync(outputPath,JSON.stringify({schema_version:2,rows:output}));
console.log(JSON.stringify({input_rows:rows.length,output_rows:output.length,output:outputPath}));

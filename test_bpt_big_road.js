"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const road=require("./bpt_big_road.js");
const cell=(s,row,col)=>s.cells[row*15+col];
assert.equal(road.ROWS,6);
assert.equal(road.COLS,15);
let s=road.build([]);
assert.equal(s.cells.length,90);
assert.equal(s.completedHands,0);
assert.equal(s.nextHand,1);
assert.equal(s.cells.filter(Boolean).length,0);
assert.throws(()=>road.build(["B","X"]),/invalid_big_road_history/);
assert.throws(()=>road.build("BPT"),/invalid_big_road_history/);
// An all-tie start has its own green mark until a B/P outcome arrives.
s=road.build(["T","T"]);
assert.deepEqual(cell(s,0,0),{side:null,ties:2,col:0,row:0});
assert.equal(s.completedHands,2);assert.equal(s.tieHands,2);
s=road.build(["T","B","T","T","B","P","T"]);
assert.equal(s.completedHands,7);
assert.equal(s.nextHand,8);
assert.equal(s.bankerHands,2);
assert.equal(s.playerHands,1);
assert.equal(s.tieHands,4);
assert.equal(cell(s,0,0).side,"B");
assert.equal(cell(s,0,0).ties,3,"initial tie and subsequent ties attach to most recent B");
assert.equal(cell(s,1,0).side,"B");
assert.equal(cell(s,1,0).ties,0,"later Banker has no tie overlay");
assert.equal(cell(s,0,1).side,"P");
assert.equal(cell(s,0,1).ties,1);
assert.equal(s.cells.filter(Boolean).length,3,"ties do not consume an independent Big Road cell");
// Different streaks start from the next top-row base column.
s=road.build(["B","B","P","P","B"]);
assert.equal(cell(s,0,0).side,"B");
assert.equal(cell(s,1,0).side,"B");
assert.equal(cell(s,0,1).side,"P");
assert.equal(cell(s,1,1).side,"P");
assert.equal(cell(s,0,2).side,"B");
// A dragon at the sixth row bends right. New streak starts at the
// next top-row base column even if a prior tail occupies row 6.
s=road.build([...Array(8).fill("B"),...Array(6).fill("P")]);
assert.equal(cell(s,5,0).side,"B");
assert.equal(cell(s,5,1).side,"B");
assert.equal(cell(s,5,2).side,"B");
for(let r=0;r<5;r++)assert.equal(cell(s,r,1).side,"P");
assert.equal(cell(s,4,2).side,"P","collision at row 6 bends streak right");
assert.equal(s.cells.filter(Boolean).length,14,"no overwritten baccarat result");
// Visible portion always remains exactly 15 columns wide, latest on the right.
s=road.build(Array.from({length:24},(_,i)=>i%2?"P":"B"));
assert.equal(s.totalRoadColumns,24);
assert.equal(s.firstVisibleColumn,9);
assert.equal(s.cells.length,90);
assert.equal(s.cells.filter(Boolean).length,15);
assert.equal(cell(s,0,0).side,"P");
assert.equal(cell(s,0,14).side,"P");
// Pure projection: undo, reload, and fresh shoe derive from the same list.
const prior=["B","P","T","B","P"];
const after=road.build(prior);
const undo=road.build(prior.slice(0,-1));
assert.equal(after.completedHands,5);
assert.equal(undo.completedHands,4);
assert.equal(undo.nextHand,5);
assert.equal(road.build(prior).completedHands,5);
assert.equal(road.build([]).completedHands,0);
// Minimal DOM harness to verify exactly 6x15 cells, title and counters.
function elem(tag,doc){
 return {tagName:tag,ownerDocument:doc,className:"",attributes:{},children:[],title:"",
  setAttribute(k,v){this.attributes[k]=v;},
  appendChild(child){this.children.push(child);return child;},
  replaceChildren(child){this.children=child.children;},
  get textContent(){return this._text||"";},
  set textContent(x){this._text=x;}
 };
}
const doc={createElement(tag){return elem(tag,this);},createDocumentFragment(){return elem("fragment",this);}};
const grid=elem("div",doc),counter=elem("span",doc);
road.render(grid,["B","T","P"],counter);
assert.equal(grid.children.length,90);
assert.equal(counter.textContent,"已開 3 局 · 下局第 4 局");
assert.equal(grid.attributes.role,"img");
assert(grid.attributes["aria-label"].includes("和1局"));
assert(grid.children[0].children[0].className.includes("bpt-road-banker"));
assert(grid.children[0].children[0].className.includes("bpt-road-has-tie"));
assert(grid.children[1].children[0].className.includes("bpt-road-player"));
road.render(grid,["B","T"],counter);
assert.equal(counter.textContent,"已開 2 局 · 下局第 3 局");
road.render(grid,[],counter);
assert.equal(counter.textContent,"已開 0 局 · 下局第 1 局");
assert.equal(grid.children.filter(x=>x.children.length).length,0);
// Wiring and style contract: projection never enters the forecast engine.
const index=fs.readFileSync("index.html","utf8");
const panel=fs.readFileSync("soft_shoe_panel.js","utf8");
const auth=fs.readFileSync("auth-client.js","utf8");
const css=fs.readFileSync("styles.css","utf8");
assert(index.includes('id="bptRoundCounter"')&&index.includes('id="bptBigRoad"'));
assert(index.includes("styles.css?v=20261106-bigroad-v1"));
assert(index.includes("auth-client.js?v=20261112-v5-1-road-aux-1"));
assert(auth.includes('"bpt_big_road.js?v=20261106-bigroad-v1"'));
assert(auth.indexOf('bpt_big_road.js?v=')<auth.indexOf('soft_shoe_panel.js?v='));
assert(panel.includes("const syncBigRoad=()=>root.BBBBigRoad.render"));
assert.equal((panel.match(/\bsyncBigRoad\(\);/g)||[]).length,3,"initial load, input and undo all synchronize");
assert(css.includes("grid-template-columns:repeat(15,minmax(0,1fr))"));
assert(css.includes("max-width:300px"));
assert(css.includes(".bpt-road-has-tie::after"));
assert(!fs.readFileSync("soft_shoe_decision_runtime.js","utf8").includes("BBBBigRoad"),
 "prediction module must not depend on road UI");
console.log("PASS Big Road 6x15, overflow, dragon tail, collision, ties, undo and compact DOM counter");

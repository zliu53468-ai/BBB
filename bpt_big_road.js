/**
 * BBB compact Baccarat Big Road (6 rows x 15 columns).
 * Display-only projection from the existing licensed B/P/T history.
 * No inference engine, betting gate, history storage, or RNG dependency.
 */
(function(root,factory){
 "use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.BBBBigRoad=api;
})(typeof window!=="undefined"?window:typeof self!=="undefined"?self:null,function(){
 "use strict";
 const ROWS=6,COLS=15;
 const valid=x=>x==="B"||x==="P"||x==="T";
 const key=(col,row)=>col+":"+row;
 function build(history){
  if(!Array.isArray(history)||history.some(x=>!valid(x)))
   throw Error("invalid_big_road_history");
  const occupied=new Map(),all=[];
  let last=null,runStartCol=-1,pendingTies=0,totalTies=0,maxCol=0;
  for(const side of history){
   if(side==="T"){
    totalTies++;
    if(last)last.ties++;
    else pendingTies++;
    continue;
   }
   let col,row;
   if(!last||last.side!==side){
    // A new Banker/Player streak begins on the top row of the next
    // base column. Previous dragon tails may occupy lower cells.
    col=last?runStartCol+1:0;row=0;
    while(occupied.has(key(col,row)))col++;
    runStartCol=col;
   }else{
    row=last.row+1;col=last.col;
    if(row>=ROWS||occupied.has(key(col,row))){
     row=last.row;col=last.col+1;
     while(occupied.has(key(col,row)))col++;
    }
   }
   const bead={col,row,side,ties:pendingTies};
   pendingTies=0;occupied.set(key(col,row),bead);all.push(bead);
   last=bead;maxCol=Math.max(maxCol,col);
  }
  const firstVisibleColumn=Math.max(0,maxCol-COLS+1);
  const cells=Array.from({length:ROWS*COLS},()=>null);
  for(const bead of all){
   const col=bead.col-firstVisibleColumn;
   if(col>=0&&col<COLS)cells[bead.row*COLS+col]={
    side:bead.side,ties:bead.ties,col:bead.col,row:bead.row
   };
  }
  if(!all.length&&pendingTies>0)
   cells[0]={side:null,ties:pendingTies,col:0,row:0};
  return Object.freeze({
   rows:ROWS,columns:COLS,cells,
   completedHands:history.length,nextHand:history.length+1,
   bankerHands:history.filter(x=>x==="B").length,
   playerHands:history.filter(x=>x==="P").length,
   tieHands:totalTies,
   firstVisibleColumn,visibleColumns:COLS,
   totalRoadColumns:all.length?maxCol+1:0,
   newest:last?{...last}:null,
   pendingTies
  });
 }
 function render(grid,history,counter){
  if(!grid||!grid.ownerDocument)throw Error("big_road_dom_missing");
  const state=build(history),doc=grid.ownerDocument;
  const frag=doc.createDocumentFragment();
  // DOM order is row-major, visual order is 15 columns x 6 rows.
  for(let row=0;row<ROWS;row++)for(let col=0;col<COLS;col++){
   const bead=state.cells[row*COLS+col],cell=doc.createElement("span");
   cell.className="bpt-road-cell";
   cell.setAttribute("aria-hidden","true");
   if(bead){
    const marker=doc.createElement("span");
    marker.className="bpt-road-bead "+
     (bead.side==="B"?"bpt-road-banker":bead.side==="P"?"bpt-road-player":"bpt-road-tie-only")+
     (bead.ties>0?" bpt-road-has-tie":"");
    cell.appendChild(marker);
    cell.title=(bead.side==="B"?"莊":bead.side==="P"?"閒":"未定方向")+
     (bead.ties?"，和局 "+bead.ties+" 次":"");
   }
   frag.appendChild(cell);
  }
  grid.replaceChildren(frag);
  grid.setAttribute("role","img");
  grid.setAttribute("aria-label",
   "百家樂大路，6列15欄。已開"+state.completedHands+"局，莊"+
   state.bankerHands+"局，閒"+state.playerHands+"局，和"+state.tieHands+
   "局。紅色莊、藍色閒、綠斜線和局。");
  if(counter)counter.textContent="已開 "+state.completedHands+" 局 · 下局第 "+state.nextHand+" 局";
  return state;
 }
 return Object.freeze({ROWS,COLS,build,render});
});

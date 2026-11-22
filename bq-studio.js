/* BQ STUDIO v1 — read-only visualization of a visitor's existing B/P/T input.
   Does NOT produce a model recommendation, access remote tables, or start a trial.
   The real BBB trialStartBtn remains owned by trial-client.js.
*/
(function(){
  "use strict";
  if(typeof document==="undefined")return;
  const history=document.getElementById("quantHistory");
  const road=document.getElementById("bqStudioRoad");
  if(!history||!road)return;
  const textByCode={B:"莊",P:"閒",T:"和"};
  const parse=raw=>{
    const parser=window.BBBQuantMath?.parseHistory;
    return typeof parser==="function"?parser(raw):[];
  };
  function sync(){
    let sequence;
    try{
      sequence=parse(history.value);
    }catch(_){
      road.setAttribute("aria-label","請檢查輸入格式，只接受莊閒和或 B/P/T");
      const warning=document.createElement("span");
      warning.className="bq-road-empty";
      warning.textContent="輸入格式有誤，請使用莊／閒／和";
      road.replaceChildren(warning);
      return;
    }
    const latest=sequence.slice(-40);
    road.setAttribute("aria-label","依照本機輸入顯示最近 "+latest.length+" 局歷史紀錄，不是下一局預測");
    if(latest.length===0){
      const empty=document.createElement("span");
      empty.className="bq-road-empty";
      empty.textContent="輸入莊／閒／和後顯示實際紀錄";
      road.replaceChildren(empty);
      return;
    }
    const fragment=document.createDocumentFragment();
    for(const code of latest){
      const bead=document.createElement("span");
      bead.className="bq-road-bead is-"+code.toLowerCase();
      bead.textContent=textByCode[code];
      bead.setAttribute("aria-hidden","true");
      fragment.append(bead);
    }
    road.replaceChildren(fragment);
  }
  history.addEventListener("input",sync);
  for(const id of ["quantAddB","quantAddP","quantAddT","quantApply","quantUndo","quantClear"]){
    document.getElementById(id)?.addEventListener("click",()=>{
      // Existing calculator's listener runs first; sync after it normalizes history.
      if(typeof queueMicrotask==="function")queueMicrotask(sync);
      else sync();
    });
  }
  const modeButtons=Array.from(document.querySelectorAll("[data-bq-studio-target]"));
  const reduced=window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  modeButtons.forEach(button=>{
    button.addEventListener("click",()=>{
      const targetId=button.dataset.bqStudioTarget;
      const card=(targetId==="quantManualCard"||targetId==="quantPreviewCard")
        ?document.getElementById(targetId):null;
      if(!card)return;
      modeButtons.forEach(other=>{
        const selected=other===button;
        other.classList.toggle("is-current",selected);
        other.setAttribute("aria-pressed",String(selected));
      });
      for(const id of ["quantManualCard","quantPreviewCard"]){
        document.getElementById(id)?.classList.toggle("bq-studio-highlight",id===targetId);
      }
      card.scrollIntoView({behavior:reduced?"auto":"smooth",block:"nearest"});
    });
  });
  sync();
})();

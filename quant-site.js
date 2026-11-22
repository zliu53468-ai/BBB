/* Shared site navigation only. Native <details> menu works without JS;
 * no dependencies on prediction/visitor/license runtimes. */
(function(){
  "use strict";
  if(typeof document==="undefined")return;
  const drawer=document.querySelector(".quant-site-mobile-drawer");
  const pathname=location.pathname.split("/").pop()||"index.html";
  const here=pathname==="index.html"?"./":pathname;
  const activeToolHash=["#quantTabProbability","#quantTabKelly","#quantTabSimulation"].includes(location.hash)
    ?location.hash:"#quantTabProbability";
  document.querySelectorAll(".quant-site-links a, .quant-site-mobile-links a").forEach(a=>{
    const relative=a.getAttribute("href")||"";
    const target=relative.split("#")[0];
    const active=(here==="./"&&relative==="./")
      ||(here==="quant-tools.html"&&relative===here+activeToolHash)
      ||(here!=="./"&&here!=="quant-tools.html"&&target===here)
      ||(here.startsWith("research-")&&target==="research-hub.html");
    if(active){
      a.setAttribute("aria-current","page");
    }
    if(drawer&&drawer.contains(a)){
      a.addEventListener("click",()=>{drawer.open=false;});
    }
  });
  if(!drawer)return;
  document.addEventListener("keydown",event=>{
    if(event.key==="Escape"&&drawer.open){drawer.open=false;drawer.querySelector("summary")?.focus();}
  });
  document.addEventListener("click",event=>{
    if(drawer.open&&!drawer.contains(event.target))drawer.open=false;
  });
})();

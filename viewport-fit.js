/**
 * BBB viewport-only layout controller. No prediction engine, results, auth,
 * or bankroll calculations are read or modified.
 * Natural app height is measured before presentation scale is applied,
 * so controls remain within one visible viewport instead of being clipped.
 */
(function(){
  "use strict";
  const app=document.getElementById("predictionApp");
  const content=document.getElementById("appViewportContent");
  if(!app||!content)return;

  let pending=false;
  function fit(){
    pending=false;
    const visible=!app.hidden;
    document.documentElement.classList.toggle("bbb-viewport-locked",visible);
    document.body.classList.toggle("bbb-viewport-locked",visible);
    if(!visible){
      content.style.removeProperty("--bbb-fit-scale");
      return;
    }
    const vh=window.visualViewport?.height||window.innerHeight||
      document.documentElement.clientHeight;
    const height=Math.max(content.scrollHeight,content.offsetHeight,1);
    const available=Math.max(1,vh-12);
    const scale=Math.min(1,available/height);
    content.style.setProperty("--bbb-fit-scale",String(scale));
  }
  function schedule(){
    if(pending)return;
    pending=true;
    window.requestAnimationFrame(fit);
  }

  if(typeof MutationObserver==="function"){
    const visibility=new MutationObserver(schedule);
    visibility.observe(app,{attributes:true,attributeFilter:["hidden"]});
    const changes=new MutationObserver(schedule);
    changes.observe(content,{subtree:true,childList:true,characterData:true});
  }
  if(typeof ResizeObserver==="function"){
    const sizes=new ResizeObserver(schedule);
    sizes.observe(content);
  }
  window.addEventListener("resize",schedule,{passive:true});
  window.visualViewport?.addEventListener("resize",schedule,{passive:true});
  document.addEventListener("toggle",schedule,true);
  window.addEventListener("load",schedule);
  if(document.fonts?.ready)document.fonts.ready.then(schedule);
  schedule();
})();

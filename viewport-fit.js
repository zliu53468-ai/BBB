/**
 * Fixed mobile viewport, no transform scaling and no resize observers.
 * Monitor ONLY auth visibility. CSS handles small screens with internal
 * scrolling instead of shrinking the entire page on toolbar changes.
 */
(function(){
  "use strict";
  const app=document.getElementById("predictionApp");
  if(!app)return;
  function syncVisibility(){
    const active=!app.hidden;
    document.documentElement.classList.toggle("bbb-viewport-locked",active);
    document.body.classList.toggle("bbb-viewport-locked",active);
  }
  syncVisibility();
  if(typeof MutationObserver==="function")
    new MutationObserver(syncVisibility).observe(app,{
      attributes:true,attributeFilter:["hidden"]
    });
})();

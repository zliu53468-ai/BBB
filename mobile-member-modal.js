/* Preserve legacy license activation while allowing a compact mobile overlay. */
(function(){
"use strict";
const panel=document.getElementById("licenseMemberPanel");
const close=document.getElementById("bbbMobileMemberClose");
const trigger=document.getElementById("memberAccessToggle");
if(!panel||!close)return;
function dismiss(){
 panel.hidden=true;
 trigger?.setAttribute("aria-expanded","false");
 (trigger||document.querySelector(".homepage-member-entry"))?.focus?.();
}
close.addEventListener("click",dismiss);
panel.addEventListener("keydown",e=>{
 if(e.key==="Escape"&&!panel.hidden){e.preventDefault();dismiss();}
});
})();

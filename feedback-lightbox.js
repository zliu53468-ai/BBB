/* Zoomable member-supplied screenshot sprite, with keyboard/touch navigation. */
(function(){
"use strict";
const dialog=document.getElementById("bbbFeedbackLightbox");
const viewport=document.getElementById("bbbFeedbackZoomViewport");
const count=document.getElementById("bbbFeedbackZoomIndex");
const close=document.getElementById("bbbFeedbackZoomClose");
const prev=document.getElementById("bbbFeedbackZoomPrev");
const next=document.getElementById("bbbFeedbackZoomNext");
if(!dialog||!viewport||!count||!close||!prev||!next)return;
let current=1,opener=null,touchX=null;
function render(i){
 current=((i%5)+5)%5+1;
 viewport.className="bbb-feedback-zoom-viewport feedback-zoom-"+current;
 viewport.setAttribute("aria-label","會員回饋對話及遊戲報表截圖，第 "+current+" 張，共 5 張");
 count.textContent=current+" / 5";
}
function open(i,source){
 opener=source;
 render(i-1);
 if(typeof dialog.showModal==="function")dialog.showModal();
 else{dialog.hidden=false;dialog.setAttribute("open","");}
}
function dismiss(){
 if(typeof dialog.close==="function"&&dialog.open)dialog.close();
 else{dialog.hidden=true;dialog.removeAttribute("open");}
 opener?.focus?.();
}
for(const button of document.querySelectorAll("[data-feedback-index]")){
 button.addEventListener("click",()=>open(Number(button.dataset.feedbackIndex)||1,button));
}
close.addEventListener("click",dismiss);
prev.addEventListener("click",()=>render(current-2));
next.addEventListener("click",()=>render(current));
dialog.addEventListener("click",event=>{if(event.target===dialog)dismiss();});
dialog.addEventListener("keydown",event=>{
 if(event.key==="ArrowRight"){event.preventDefault();render(current);}
 if(event.key==="ArrowLeft"){event.preventDefault();render(current-2);}
});
dialog.addEventListener("close",()=>opener?.focus?.());
viewport.addEventListener("touchstart",event=>{
 if(event.touches.length===1)touchX=event.touches[0].clientX;
},{passive:true});
viewport.addEventListener("touchend",event=>{
 if(touchX===null||event.changedTouches.length!==1)return;
 const dx=event.changedTouches[0].clientX-touchX;
 touchX=null;
 if(Math.abs(dx)>45)render(current+(dx>0?-2:0));
},{passive:true});
render(0);
})();

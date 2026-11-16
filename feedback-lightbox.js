/* Zoomable member-supplied screenshot sprite, with keyboard/touch navigation. */
(function(){
"use strict";
const dialog=document.getElementById("bbbFeedbackLightbox");
const viewport=document.getElementById("bbbFeedbackZoomViewport");
const image=document.getElementById("bbbFeedbackZoomImg");
const original=document.getElementById("bbbFeedbackZoomOriginal");
const images=[
 {src:"assets/member-feedback-01.webp",w:1280,h:1066},
 {src:"assets/member-feedback-02.webp",w:647,h:627},
 {src:"assets/member-feedback-03.webp",w:1257,h:1079},
 {src:"assets/member-feedback-04.webp",w:1017,h:939},
 {src:"assets/member-feedback-05.webp",w:762,h:755}
];

const count=document.getElementById("bbbFeedbackZoomIndex");
const close=document.getElementById("bbbFeedbackZoomClose");
const prev=document.getElementById("bbbFeedbackZoomPrev");
const next=document.getElementById("bbbFeedbackZoomNext");
if(!dialog||!viewport||!image||!original||!count||!close||!prev||!next)return;
let current=1,opener=null,touchX=null;
function render(i){
 current=((i%5)+5)%5+1;
 viewport.className="bbb-feedback-zoom-viewport feedback-zoom-"+current;
 const selected=images[current-1];
 image.src=selected.src;
 image.width=selected.w;
 image.height=selected.h;
 image.alt="會員回饋原始截圖 "+current+"，共 5 張";
 original.href=selected.src;
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

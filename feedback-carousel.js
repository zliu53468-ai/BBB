/* One-screen member feedback gallery: five original figures, one mobile slide.
   Desktop keeps the regular grid. This module does not touch prediction paths. */
(function(){
"use strict";
const gallery=document.getElementById("bbbFeedbackCarousel");
const prev=document.getElementById("bbbFeedbackCardPrev");
const next=document.getElementById("bbbFeedbackCardNext");
const counter=document.getElementById("bbbFeedbackCardCounter");
if(!gallery||!prev||!next||!counter)return;
const cards=Array.from(gallery.querySelectorAll(".member-feedback-card"));
if(cards.length!==5)return;
let active=0;
function mobile(){return typeof window.matchMedia==="function"&&window.matchMedia("(max-width:760px)").matches;}
function refresh(){
 const compact=mobile();
 cards.forEach((card,i)=>{card.hidden=compact&&i!==active;});
 counter.textContent=(active+1)+" / "+cards.length;
 prev.disabled=!compact;next.disabled=!compact;
}
function move(step){active=(active+step+cards.length)%cards.length;refresh();}
prev.addEventListener("click",()=>move(-1));
next.addEventListener("click",()=>move(1));
window.addEventListener("resize",refresh);
refresh();
})();

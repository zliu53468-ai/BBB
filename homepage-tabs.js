/* 3 accessible content tabs; content stays in HTML for navigation and indexing. */
(function(){
"use strict";
const buttons=[
 document.getElementById("bbbTabBtnExperience"),
 document.getElementById("bbbTabBtnFeedback"),
 document.getElementById("bbbTabBtnLearn")
];
const panels=[
 document.getElementById("bbbTabExperience"),
 document.getElementById("bbbTabFeedback"),
 document.getElementById("bbbTabLearn")
];
if(buttons.some(x=>!x)||panels.some(x=>!x))return;
function select(index,{focus=false}={}){
 if(index<0||index>=buttons.length)return;
 buttons.forEach((button,i)=>{
   const selected=i===index;
   button.setAttribute("aria-selected",String(selected));
   button.tabIndex=selected?0:-1;
   panels[i].hidden=!selected;
 });
 if(focus)buttons[index].focus();
}
buttons.forEach((button,i)=>{
 button.addEventListener("click",()=>select(i));
 button.addEventListener("keydown",event=>{
   const key=event.key;
   const j=key==="ArrowRight"?(i+1)%3
     :key==="ArrowLeft"?(i+2)%3
     :key==="Home"?0:key==="End"?2:-1;
   if(j<0)return;
   event.preventDefault();
   select(j,{focus:true});
 });
});
select(0);
})();

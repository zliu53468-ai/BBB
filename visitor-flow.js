/* Visitor entry and first-party funnel feedback.
 * Paid license validation stays entirely in the untouched auth-client.js.
 * This stores no personal identifiers and sends no analytics or registration data.
 */
(function(){
  "use strict";
  const followup=document.getElementById("returnVisitorCard");
  const member=document.getElementById("licenseMemberPanel");
  const showMember=document.getElementById("memberAccessToggle");
  const reason=document.getElementById("returnVisitorReason");
  if(!followup||!member)return;
  const key="bbb-funnel-last-action-v1";
  function readAction(){
    let payload=null;
    try{payload=localStorage.getItem(key)||sessionStorage.getItem(key);}catch(_){
      try{payload=sessionStorage.getItem(key);}catch(_){}
    }
    if(!payload)return null;
    try{
      const info=JSON.parse(payload);
      if(!["casino","line"].includes(info.action)||!Number.isFinite(info.at))return null;
      if(Date.now()-info.at>30*24*60*60*1000||Date.now()<info.at)return null;
      return info.action;
    }catch(_){return null;}
  }
  function revealMember(){
    member.hidden=false;
    showMember?.setAttribute("aria-expanded","true");
    if(new URLSearchParams(location.search).get("license")==="1")return;
    member.scrollIntoView?.({behavior:"smooth",block:"nearest"});
  }
  showMember?.addEventListener("click",revealMember);
  const action=readAction();
  if(action){
    followup.hidden=false;
    if(reason)reason.textContent=action==="casino"
      ?"你剛才曾點開合作娛樂城的連結。若需要領取 BBB 程式開通碼，請聯繫官方管理員。"
      :"你曾點過官方 LINE 聯繫入口。若要領取開通碼或詢問後續使用方式，請再聯繫官方管理員。";
  }
  if(new URLSearchParams(location.search).get("license")==="1"){
    followup.hidden=false;
    revealMember();
  }
  window.addEventListener("pageshow",()=>{
    if(readAction())followup.hidden=false;
  });
})();

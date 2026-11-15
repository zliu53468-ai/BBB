// Safe outbound links plus one-per-page random casino choice.
// The actual casino registration page is opened by a user-clicked native link in
// a NEW TAB; the account/transfer prompt remains on BBB's own tab.
(function(){
  "use strict";
  const options=window.BBB_SITE_LINKS||{};
  const lastKey="bbb-last-casino-link-v1";
  let lastOpenedInPage="";
  function safeUrl(value){
    if(typeof value!=="string"||!value.trim())return null;
    try {
      const u=new URL(value.trim());
      return u.protocol==="https:"&&u.hostname?u.href:null;
    } catch(_){return null;}
  }
  function randomIndex(count){
    if(count<=1)return 0;
    // Rejection sampling avoids modulo bias and gives every partner equal weight.
    if(window.crypto?.getRandomValues){
      const n=new Uint32Array(1);
      const max=4294967296;
      const limit=max-(max%count);
      do{window.crypto.getRandomValues(n);}while(n[0]>=limit);
      return n[0]%count;
    }
    return Math.floor(Math.random()*count);
  }
  function previous(){
    try{return lastOpenedInPage||window.sessionStorage.getItem(lastKey)||"";}catch(_){return lastOpenedInPage;}
  }
  function remember(url){
    lastOpenedInPage=url;
    try{window.sessionStorage.setItem(lastKey,url);}catch(_){}
  }
  function pickCasino(){
    const entries=Array.isArray(options.casinoRegisterUrls)
      ?options.casinoRegisterUrls.map(safeUrl).filter(Boolean)
      :[safeUrl(options.casinoRegisterUrl)].filter(Boolean);
    const urls=[...new Set(entries)];
    if(!urls.length)return null;
    // If the same browser returns during this session, avoid the last opened site.
    // Other visitors are independently assigned; absolute uniqueness is impossible
    // with five options, but the first assignment is uniformly random.
    const previousUrl=previous();
    const pool=urls.length>1?urls.filter(url=>url!==previousUrl):urls;
    return pool[randomIndex(pool.length)];
  }
  // The initial href is a safe fallback for native new-tab navigation.
  // A fresh destination is selected synchronously on EACH user click.
  const assignedCasino=pickCasino();
  function openAccountPrompt(){
    const dialog=document.getElementById("casinoAccountDialog");
    if(!dialog)return;
    if(typeof dialog.showModal==="function"){
      if(!dialog.open)dialog.showModal();
    }else{
      dialog.hidden=false;
      dialog.setAttribute("open","");
    }
  }
  function closeAccountPrompt(){
    const dialog=document.getElementById("casinoAccountDialog");
    if(!dialog)return;
    if(typeof dialog.close==="function")dialog.close();
    else{dialog.hidden=true;dialog.removeAttribute("open");}
  }
  document.getElementById("casinoAccountClose")?.addEventListener("click",closeAccountPrompt);
  document.getElementById("casinoAccountDismiss")?.addEventListener("click",closeAccountPrompt);
  for(const el of document.querySelectorAll("[data-bbb-link]")){
    const key=el.getAttribute("data-bbb-link");
    const href=key==="casinoRegisterUrl"?assignedCasino:safeUrl(options[key]);
    if(href){
      el.href=href;
      el.target="_blank";
      el.rel="noopener noreferrer"+(key==="lineContactUrl"?"":" sponsored");
      el.removeAttribute("aria-disabled");
      el.classList.remove("outbound-disabled");
      if(key==="casinoRegisterUrl"){
        el.addEventListener("click",()=>{
          // This was previously fixed at page-load, which repeated the same
          // destination on every click in the same page. Reselect now.
          const destination=pickCasino();
          if(!destination)return;
          el.href=destination;
          remember(destination);
          // Keep native navigation (target=_blank) and the BBB dialog.
          openAccountPrompt();
        });
      }
    }else{
      el.removeAttribute("href");
      el.setAttribute("aria-disabled","true");
      el.classList.add("outbound-disabled");
      el.title="管理員尚未設定此連結";
      el.addEventListener("click",event=>event.preventDefault());
    }
  }
})();

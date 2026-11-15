// Safe outbound links plus one-per-page random casino choice.
// The actual casino registration page is opened by a user-clicked native link in
// a NEW TAB; the account/transfer prompt remains on BBB's own tab.
(function(){
  "use strict";
  const options=window.BBB_SITE_LINKS||{};
  const lastKey="bbb-last-casino-link-v1";
  const funnelKey="bbb-funnel-last-action-v1";
  const nextHint=document.getElementById("casinoNextHint");
  let casinoButton=null;
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
  function markFunnel(action){
    if(action!=="casino"&&action!=="line")return;
    const payload=JSON.stringify({action,at:Date.now()});
    // A first-party client-side signal only: no user ID, cookies or API calls.
    try{window.localStorage.setItem(funnelKey,payload);}catch(_){
      try{window.sessionStorage.setItem(funnelKey,payload);}catch(_){}
    }
  }
  function prepareNextCasino(){
    if(!casinoButton)return;
    const next=pickCasino();
    if(!next)return;
    casinoButton.href=next;
    if(nextHint)nextHint.hidden=false;
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
    prepareNextCasino();
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
        casinoButton=el;
        el.addEventListener("click",()=>{
          // Native target=_blank uses this destination from the current click.
          const destination=pickCasino();
          if(!destination)return;
          el.href=destination;
          remember(destination);
          markFunnel("casino");
          openAccountPrompt();
          // A new tab may suspend this one. When the visitor returns, the
          // NEXT destination is already prepared, without any page refresh.
          // Never trigger another unsolicited popup or redirect loop.
          if(typeof window.setTimeout==="function")window.setTimeout(prepareNextCasino,0);
        });
      } else if(key==="lineContactUrl"){
        el.addEventListener("click",()=>markFunnel("line"));
      }
    }else{
      el.removeAttribute("href");
      el.setAttribute("aria-disabled","true");
      el.classList.add("outbound-disabled");
      el.title="管理員尚未設定此連結";
      el.addEventListener("click",event=>event.preventDefault());
    }
  }
  // Restoring from the browser back-forward cache must also refresh the
  // destination, since the entire document (including the old href) is reused.
  if(casinoButton&&typeof window.addEventListener==="function"){
    window.addEventListener("pageshow",event=>{
      if(event?.persisted&&lastOpenedInPage)prepareNextCasino();
    });
    window.addEventListener("focus",()=>{
      if(lastOpenedInPage&&typeof window.setTimeout==="function")
        window.setTimeout(prepareNextCasino,0);
    });
  }
})();

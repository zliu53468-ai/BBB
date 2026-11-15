// Shared outbound-link binder. Only user-supplied, verified HTTPS destinations work.
(function(){
  "use strict";
  const options=window.BBB_SITE_LINKS||{};
  function safeUrl(value){
    if(typeof value!=="string"||!value.trim())return null;
    try {const u=new URL(value.trim()); return u.protocol==="https:"?u.href:null;}
    catch(_){return null;}
  }
  for(const el of document.querySelectorAll("[data-bbb-link]")){
    const href=safeUrl(options[el.getAttribute("data-bbb-link")]);
    if(href){
      el.href=href;
      el.target="_blank";
      el.rel="noopener noreferrer sponsored";
      el.removeAttribute("aria-disabled");
    } else {
      el.removeAttribute("href");
      el.setAttribute("aria-disabled","true");
      el.classList.add("outbound-disabled");
      el.title="管理員尚未設定此連結";
      el.addEventListener("click",event=>event.preventDefault());
    }
  }
})();

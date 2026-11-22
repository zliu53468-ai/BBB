/* Four-page corporate navigation.
 * The four main links navigate to actual pages. Each page has a distinct,
 * source-rendered submenu. Desktop hover/focus previews category submenus;
 * mobile taps navigate normally and no full-screen dropdown is used.
 * No contact with BBB Soft Shoe or auth/trial paths.
 */
(function(){
  "use strict";
  if(typeof document==="undefined")return;
  const header=document.querySelector(".quant-corporate-header");
  if(!header)return;
  const active=header.dataset.bqCategory||"home";
  const allowed=new Set(["home","tools","research","risk"]);
  const tabs=Array.from(header.querySelectorAll("[data-bq-tab]"));
  const menus=Array.from(header.querySelectorAll("[data-bq-menu]"));
  function preview(category){
    header.dataset.bqActive=allowed.has(category)?category:active;
  }
  function reset(){preview(active);}
  tabs.forEach(tab=>{
    tab.addEventListener("pointerenter",event=>{
      if(event.pointerType==="mouse"||event.pointerType==="pen"){
        preview(tab.dataset.bqTab);
      }
    });
    tab.addEventListener("focus",()=>preview(tab.dataset.bqTab));
  });
  // Hovering the currently previewed submenu keeps it visible; leaving the
  // whole header restores the submenu associated with the actual page.
  menus.forEach(menu=>{
    menu.addEventListener("pointerenter",event=>{
      if(event.pointerType==="mouse"||event.pointerType==="pen"){
        preview(menu.dataset.bqMenu);
      }
    });
  });
  header.addEventListener("pointerleave",reset);
  header.addEventListener("focusout",event=>{
    if(!event.relatedTarget||!header.contains(event.relatedTarget))reset();
  });
  document.addEventListener("keydown",event=>{
    if(event.key==="Escape"){
      reset();
      const current=tabs.find(tab=>tab.dataset.bqTab===active);
      if(header.contains(document.activeElement))current?.focus();
    }
  });
  reset();
})();

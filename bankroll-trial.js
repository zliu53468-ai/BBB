/* BBB trial bankroll / stop-win display.
 * This UI only tracks manually entered money. It never places bets or changes
 * BBBSoftShoe, its safety gates, licensing, or the Cloudflare trial deadline.
 */
(function(){
  "use strict";
  const storageKey="bbb-trial-bankroll-target-v1";
  const dialog=document.getElementById("bbbBankrollDialog");
  const form=document.getElementById("bbbBankrollForm");
  const initialInput=document.getElementById("bbbInitialBankroll");
  const preview=document.getElementById("bbbBankrollTargetPreview");
  const error=document.getElementById("bbbBankrollError");
  const cancel=document.getElementById("bbbBankrollCancel");
  const startBtn=document.getElementById("trialStartBtn");
  const panel=document.getElementById("bbbTrialBankrollPanel");
  const updateForm=document.getElementById("bbbMoneyUpdateForm");
  const currentInput=document.getElementById("bbbMoneyUpdateAmount");
  if(!dialog||!form||!initialInput||!startBtn||!panel||!updateForm)return;
  let confirmed=false,mounted=false,initial=0,current=0,hit=false,watch=null,stopWinTimer=null;

  function money(n){
    return Number.isFinite(n)?"NT$ "+n.toLocaleString("zh-TW",{maximumFractionDigits:2,minimumFractionDigits:0}):"—";
  }
  function amount(v,{zero=false}={}){
    const n=Number(v);
    if(!Number.isFinite(n)||n>(1e9)||(zero?n<0:n<=0))return null;
    return Math.round(n*100)/100;
  }
  function read(){
    try{
      const data=JSON.parse(sessionStorage.getItem(storageKey)||"null");
      if(!data)return null;
      const capital=amount(data.initial),balance=amount(data.current,{zero:true});
      if(capital===null||balance===null)return null;
      return {initial:capital,current:balance,hit:data.hit===true};
    }catch(_){return null;}
  }
  function save(){
    try{sessionStorage.setItem(storageKey,JSON.stringify({initial,current,hit}));}catch(_){}
  }
  function showDialog(){
    if(typeof dialog.showModal==="function"){
      if(!dialog.open)dialog.showModal();
    }else{dialog.hidden=false;dialog.setAttribute("open","");}
    initialInput.focus?.();
  }
  function closeDialog(){
    if(typeof dialog.close==="function"&&dialog.open)dialog.close();
    else{dialog.hidden=true;dialog.removeAttribute("open");}
  }
  function promptAndStart(){
    confirmed=false;
    const last=read();
    if(last)initialInput.value=String(last.initial);
    renderPreview();
    if(error)error.textContent="";
    showDialog();
  }
  function renderPreview(){
    const number=amount(initialInput.value);
    if(preview)preview.textContent=number===null?"—":money(number*1.6);
  }
  function updateText(id,text){
    const node=document.getElementById(id);
    if(node)node.textContent=text;
  }
  function scheduleStopWinRedirect(){
    if(!mounted||!hit||stopWinTimer!==null)return;
    stopWinTimer=setTimeout(()=>{
      if(!mounted||!hit)return;
      if(watch!==null){clearInterval(watch);watch=null;}
      // User-entered current balance, not a claimed prediction success rate.
      // Reuse the existing referral and LINE trial-end landing page.
      window.location.replace("./trial-ended.html?reason=profit60");
    },1800);
  }
  function updateKelly(){
    if(!mounted||!initial)return;
    const target=Math.round(initial*1.6*100)/100;
    const goalProfit=Math.round(initial*.6*100)/100;
    if(current>=target-0.005)hit=true;
    if(hit){save();scheduleStopWinRedirect();}
    updateText("bbbMoneyStart",money(initial));
    updateText("bbbMoneyTarget",money(target));
    updateText("bbbMoneyCurrent",money(current));
    const progress=Math.max(0,Math.min(100,Math.round((current-initial)/Math.max(0.01,goalProfit)*100)));
    const progressBar=document.querySelector(".bbb-profit-progress");
    const progressFill=document.getElementById("bbbMoneyProgress");
    if(progressBar)progressBar.setAttribute("aria-valuenow",String(progress));
    if(progressFill)progressFill.style.width=progress+"%";
    panel.classList.toggle("target-hit",hit);
    if(hit){
      updateText("bbbMoneyKelly","已止盈");
      updateText("bbbMoneyStake",money(0));
      updateText("bbbMoneyStatus","已達起始本金盈利 60% 的止盈目標，正在前往試用結束／官方管理員及娛樂城頁面。本次試用不再提供參考下注金額。");
      return;
    }
    const report=window.__BBB_SOFT_SHOE_PANEL__?.latestReport;
    const side=report?.direction;
    const wagerOk=report?.validated_edge===true&&
      (side==="B"||side==="P")&&
      (report.action==="Bet Banker"||report.action==="Bet Player");
    const f=wagerOk&&Number.isFinite(report.bankroll_fraction)?
      Math.max(0,Math.min(1,report.bankroll_fraction)):0;
    if(f<=0){
      updateText("bbbMoneyKelly","0% · 觀望");
      updateText("bbbMoneyStake",money(0));
      updateText("bbbMoneyStatus","目前未取得符合模型閘門的正向訊號，分數凱莉參考比例為 0%。已手動記錄目前資金："+money(current)+"，止盈目標："+money(target)+"。");
      return;
    }
    updateText("bbbMoneyKelly",(f*100).toFixed(2)+"%（模型分數凱莉）");
    updateText("bbbMoneyStake",money(Math.round(current*f*100)/100));
    updateText("bbbMoneyStatus","依原預測引擎的安全閘門和分數凱莉比例估算；參考額使用手動輸入的目前資金，不會自動下注或調高模型門檻。");
  }
  function onTrialMounted(){
    const previous=read();
    if(!previous)return;
    initial=previous.initial;current=previous.current;hit=previous.hit;
    mounted=true;
    panel.hidden=false;
    if(currentInput)currentInput.value=String(current);
    if(watch!==null)clearInterval(watch);
    watch=setInterval(updateKelly,1200);
    updateKelly();
  }
  initialInput.addEventListener("input",renderPreview);
  cancel?.addEventListener("click",closeDialog);
  form.addEventListener("submit",event=>{
    event.preventDefault();
    const capital=amount(initialInput.value);
    if(capital===null){
      if(error)error.textContent="請輸入有效的正數本金（最高 NT$ 10 億）。";
      return;
    }
    if(stopWinTimer!==null){clearTimeout(stopWinTimer);stopWinTimer=null;}
    initial=capital;current=capital;hit=false;confirmed=true;
    save();closeDialog();
    // The next click is a local synthetic click to the existing BBB trial
    // handler. No DG navigation or independent betting engine is involved.
    startBtn.click();
  });
  updateForm.addEventListener("submit",event=>{
    event.preventDefault();
    if(!mounted)return;
    const balance=amount(currentInput.value,{zero:true});
    if(balance===null){
      updateText("bbbMoneyStatus","請輸入有效的目前資金（可以是 0）。");
      return;
    }
    current=balance;
    save();updateKelly();
  });
  window.BBB_BANKROLL={
    promptAndStart,
    consumeConfirmation(){
      if(!confirmed)return false;
      confirmed=false;return true;
    },
    onTrialMounted,
    getSnapshot(){return mounted?{initial,current,target:Math.round(initial*1.6*100)/100,hit}:null;}
  };
})();

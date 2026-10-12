/* BBB-only browser adaptation of BGS road_detector.py.
 * Original source: zliu53468-ai/BGS0.3V.1/road_detector.py (read-only).
 * BGS Python OpenCV/EasyOCR cannot run inside GitHub Pages. This is a bounded
 * Canvas HSV 6-row big-road port, NOT the full Python OCR / venue detector.
 * No image is uploaded to a server. Low-confidence roads require recrop.
 */
(function(root){
"use strict";
const el=id=>document.getElementById(id), HISTORY_KEY="bgs256d_frozen_6x15_forward_v18";
const ROIS={
  dg_mobile:[.302,.660,.653,.104],dg_mobile_low:[.302,.700,.653,.104],
  dg_browser:[.330,.730,.650,.092],dg_android:[.302,.708,.653,.130],
  dg_compact:[.305,.710,.685,.125],dg_desktop:[.240,.803,.145,.135],
  mt_mobile:[.315,.733,.630,.080],mt_desktop:[.619791667,.716500554,.201388889,.148394241],
  db_mobile:[.360,.755,.640,.075],db_desktop:[.58,.72,.40,.18],
  db_desktop_alt:[.55,.70,.43,.20],wide:[.265,0,.735,.64],
  generic:[0,.58,1,.42],mobile_fallback:[.280,.670,.680,.200],full:[0,0,1,1]
};
const MAX_SIDE=1600,MAX_RECONSTRUCT_NODES=18000,RECONSTRUCT_MS=55;
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
let sourceCanvas=null,preview=null,selection=null,drag=null,proposal=null,busy=false,scanNo=0;
const msg=(str,error=false)=>{const e=el("bbbScanMessage");if(e){e.textContent=str;e.classList.toggle("warning",!!error);}};
const history=()=>{try{return (JSON.parse(localStorage.getItem(HISTORY_KEY)||"null")?.history||[]).filter(x=>["B","P","T"].includes(x));}catch(_){return [];}};
function readBitmap(file){
  return new Promise((resolve,reject)=>{
    if(!file||!/^image\//.test(file.type))return reject(Error("請選擇圖片檔"));
    if(file.size>16*1024*1024)return reject(Error("圖片超過 16MB，請先縮小"));
    const url=URL.createObjectURL(file),img=new Image();
    img.onload=()=>{URL.revokeObjectURL(url);const s=Math.min(1,MAX_SIDE/Math.max(img.naturalWidth,img.naturalHeight));
      const c=document.createElement("canvas");c.width=Math.round(img.naturalWidth*s);c.height=Math.round(img.naturalHeight*s);
      if(c.width<30||c.height<30)return reject(Error("圖片尺寸不足"));
      const ctx=c.getContext("2d",{willReadFrequently:true});ctx.drawImage(img,0,0,c.width,c.height);resolve(c);};
    img.onerror=()=>{URL.revokeObjectURL(url);reject(Error("此圖片格式無法讀取，請改用 JPG 或 PNG"));};
    img.src=url;
  });
}
function showCanvas(){
  if(!sourceCanvas)return;
  const c=el("bbbScanCanvas"),maxW=Math.max(250,Math.min(800,el("bbbScanVisual").clientWidth||600));
  const scale=Math.min(1,maxW/sourceCanvas.width,430/sourceCanvas.height);
  c.width=Math.max(1,Math.round(sourceCanvas.width*scale));
  c.height=Math.max(1,Math.round(sourceCanvas.height*scale));
  const ctx=c.getContext("2d");ctx.drawImage(sourceCanvas,0,0,c.width,c.height);
  const s=selection;if(s){
    const x=s[0]*c.width,y=s[1]*c.height,w=s[2]*c.width,h=s[3]*c.height;
    ctx.fillStyle="rgba(109,69,230,.17)";ctx.fillRect(x,y,w,h);
    ctx.strokeStyle="#f6dfff";ctx.lineWidth=2;ctx.strokeRect(x,y,w,h);
  }
  el("bbbScanVisual").hidden=false;
}
function hslMask(r,g,b){
  const max=Math.max(r,g,b),min=Math.min(r,g,b),dif=max-min;
  if(max<42||dif<Math.max(22,max*.19))return 0;
  let h=0;
  if(dif){
    if(max===r)h=((g-b)/dif+6)%6;
    else if(max===g)h=(b-r)/dif+2;
    else h=(r-g)/dif+4;
    h*=30; // OpenCV HSV hue 0..180.
  }
  if(h<=18||h>=163)return 1; // BGS red
  if(h>=86&&h<=145)return 2; // BGS blue
  if(h>=33&&h<=91)return 4; // tie green
  return 0;
}
function reconstruct(cells){
  const grid=new Map(cells.filter(x=>x.outcome).map(x=>[x.col+","+x.row,x]));
  if(!grid.size||!grid.has("0,0"))return null;
  let count=0,solutions=0,sol=null,deadline=performance.now()+RECONSTRUCT_MS,aborted=false;
  let best=0;const used=new Set(["0,0"]),route=[grid.get("0,0")];
  function walk(cur,runCol,tail){
    if(++count>MAX_RECONSTRUCT_NODES||performance.now()>deadline){aborted=true;return;}
    if(route.length>best)best=route.length;
    if(route.length===grid.size){solutions++;sol=route.slice();return;}
    if(solutions>=2||aborted)return;
    const choices=[];
    let [cx,cy]=[cur.col,cur.row];
    let nx,ny,nextTail;
    if(!tail&&cy<5&&!used.has(cx+","+(cy+1))){nx=cx;ny=cy+1;nextTail=false;}
    else{nx=cx+1;ny=cy;while(used.has(nx+","+ny))nx++;nextTail=true;}
    const same=grid.get(nx+","+ny);
    if(same&&same.outcome===cur.outcome&&!used.has(nx+","+ny))choices.push([same,runCol,nextTail]);
    let start=runCol+1;while(used.has(start+",0"))start++;
    const other=grid.get(start+",0");
    if(other&&other.outcome!==cur.outcome&&!used.has(start+",0"))choices.push([other,start,false]);
    for(const [node,newRun,newTail] of choices){
      const key=node.col+","+node.row;used.add(key);route.push(node);
      walk(node,newRun,newTail);route.pop();used.delete(key);
      if(solutions>=2||aborted)break;
    }
  }
  walk(route[0],0,false);
  if(solutions!==1||aborted)return null;
  return sol;
}
function scoreRegion(roi,columns){
  const x=Math.floor(roi[0]*sourceCanvas.width),y=Math.floor(roi[1]*sourceCanvas.height),
    w=Math.floor(roi[2]*sourceCanvas.width),h=Math.floor(roi[3]*sourceCanvas.height);
  if(w<columns*3||h<36)return null;
  const ctx=sourceCanvas.getContext("2d",{willReadFrequently:true}),raw=ctx.getImageData(x,y,w,h).data;
  const cells=[],total=columns*6;let uncertain=0;
  for(let row=0;row<6;row++)for(let col=0;col<columns;col++){
    const x1=Math.floor(col*w/columns),x2=Math.floor((col+1)*w/columns),
      y1=Math.floor(row*h/6),y2=Math.floor((row+1)*h/6);
    const dx=x2-x1,dy=y2-y1;
    const mx=Math.max(1,Math.round(dx*.12)),my=Math.max(1,Math.round(dy*.12));
    let b=0,p=0,t=0;
    // Area-sampled BGS HSV color mask, ignoring grid boundaries.
    for(let py=y1+my;py<y2-my;py++)for(let px=x1+mx;px<x2-mx;px++){
      const o=(py*w+px)*4,type=hslMask(raw[o],raw[o+1],raw[o+2]);
      if(type===1)b++;else if(type===2)p++;else if(type===4)t++;
    }
    const area=Math.max(1,(dx-2*mx)*(dy-2*my)),dom=Math.max(b,p),secondary=Math.min(b,p);
    const minimum=Math.max(4,Math.round(area*.018));
    let outcome="",unsure=false;
    if(dom>=minimum&&dom/(secondary+1)>=1.25)outcome=b>p?"B":"P";
    else if(dom>=minimum*.75)unsure=true;
    const tie=outcome&&t>=Math.max(7,area*.012);
    const confidence=outcome?clamp(.53*Math.min(1,dom/(minimum*2))+.47*(dom-secondary)/Math.max(1,dom+secondary),0,1):0;
    if(unsure)uncertain++;
    cells.push({col,row,outcome,unsure,tie,confidence,b,p});
  }
  const filled=cells.filter(x=>x.outcome);
  if(filled.length<5||uncertain>Math.max(2,filled.length*.12))return null;
  const firstCol=Math.min(...filled.map(x=>x.col));
  // BGS reconstruction requires first visible shoe column to start at 0.
  const normalized=filled.map(x=>({...x,col:x.col-firstCol}));
  const route=reconstruct(normalized);if(!route)return null;
  const avg=filled.reduce((a,x)=>a+x.confidence,0)/filled.length;
  if(avg<.48)return null;
  const outcomes=[];
  for(const cell of route){outcomes.push(cell.outcome);if(cell.tie)outcomes.push("T");}
  if(outcomes.length>180)return null;
  return {outcomes,quality:avg,recognized:filled.length,uncertain,
    tieMarkers:route.filter(x=>x.tie).length,columns,roi,
    score:filled.length*(.56+avg*.44)-uncertain*3};
}
function candidates(){
  if(selection)return [selection];
  const mode=el("bbbScanProfile").value;
  const names=mode==="auto"?
    (sourceCanvas.height>sourceCanvas.width?
      ["dg_mobile","dg_mobile_low","dg_browser","dg_android","dg_compact","mt_mobile","db_mobile","mobile_fallback","full"]:
      ["mt_desktop","db_desktop","db_desktop_alt","dg_desktop","wide","generic","full"]):
    mode==="crop"?["full"]:
    mode==="DG"?["dg_mobile","dg_mobile_low","dg_browser","dg_android","dg_desktop","dg_compact","wide"]:
    mode==="MT"?["mt_mobile","mt_desktop","wide"]:
    ["db_mobile","db_desktop","db_desktop_alt","wide"];
  return names.map(x=>ROIS[x]);
}
// BGS Python searches near its fixed ROI instead of assuming screenshot
// browser bars/camera crop always align exactly. Keep only four cheap bounded
// variants per ROI to avoid multiscanning huge images on mobile.
function roiVariants(roi){
  if(selection)return [roi];
  const [x,y,w,h]=roi;
  return [roi,
    [x,y+.08*h,w,.82*h],
    [x,y,w,.82*h],
    [x,y+.18*h,w,.82*h]
  ].filter(r=>r[0]>=0&&r[1]>=0&&r[0]+r[2]<=1&&r[1]+r[3]<=1);
}
function showSelectedRoad(roi,columns){
  const target=el("bbbScanRoadCrop");if(!target||!sourceCanvas)return;
  const [x,y,w,h]=roi;
  const tw=Math.min(900,Math.max(100,Math.round(w*sourceCanvas.width)));
  const th=Math.max(35,Math.round(tw*h*sourceCanvas.height/(w*sourceCanvas.width)));
  target.width=tw;target.height=th;
  const ctx=target.getContext("2d");
  ctx.drawImage(sourceCanvas,x*sourceCanvas.width,y*sourceCanvas.height,
    w*sourceCanvas.width,h*sourceCanvas.height,0,0,tw,th);
  ctx.strokeStyle="rgba(170,112,255,.5)";ctx.lineWidth=1;
  for(let r=1;r<6;r++){const yy=Math.round(r*th/6)+.5;
    ctx.beginPath();ctx.moveTo(0,yy);ctx.lineTo(tw,yy);ctx.stroke();}
  for(let c=1;c<columns;c++){const xx=Math.round(c*tw/columns)+.5;
    ctx.beginPath();ctx.moveTo(xx,0);ctx.lineTo(xx,th);ctx.stroke();}
}
function scan(){
  if(!sourceCanvas)return;
  proposal=null;el("bbbScanConfirm").disabled=true;
  msg("正在依照 BGS 六列大路辨識規則分析截圖…");
  let best=null;
  for(const base of candidates())for(const roi of roiVariants(base)){
    const aspect=roi[2]*sourceCanvas.width/(roi[3]*sourceCanvas.height);
    const estimate=clamp(Math.round(aspect*6),8,48);
    const cols=[estimate,estimate-1,estimate+1,15].filter((x,i,a)=>x>=8&&x<=48&&a.indexOf(x)===i);
    for(const n of cols){
      const result=scoreRegion(roi,n);
      if(result&&(!best||result.score>best.score))best=result;
    }
  }
  if(!best){msg("無法可信地辨識完整六列大路。請在圖片上拖曳框選真正的「大路」格線區，再按重新辨識；不會寫入任何牌局。",true);el("bbbScanResult").hidden=true;return;}
  proposal=best.outcomes.slice();
  el("bbbScanResult").hidden=false;
  el("bbbScanSummary").textContent="辨識 "+proposal.length+" 局（紅／藍格 "+best.recognized+" 個），圖像品質指標 "+Math.round(best.quality*100)+"/100（非預測機率）。";
  showSelectedRoad(best.roi,best.columns);
  el("bbbScanTieNote").textContent=best.tieMarkers?
    "偵測到 "+best.tieMarkers+" 個綠色和局標記：多次和局的數字不一定能辨識，請核對。":"若原圖含和局標記但未顯示在歷史預覽，請重新裁圖核對。";
  updateTokens();
  el("bbbScanConfirm").disabled=false;
  msg("已辨識完成。請核對下方順序，確認後才會加入歷史。");
}
function updateTokens(){
  const box=el("bbbScanTokens");box.replaceChildren();
  for(let i=0;i<proposal.length;i++){
    const button=document.createElement("button");button.type="button";
    button.className="bbb-scan-token "+proposal[i];
    button.textContent=proposal[i]==="B"?"莊":proposal[i]==="P"?"閒":"和";
    button.title="第 "+(i+1)+" 局，點擊可切換莊／閒／和";
    button.addEventListener("click",()=>{
      proposal[i]=proposal[i]==="B"?"P":proposal[i]==="P"?"T":"B";
      updateTokens();msg("已手動校正辨識預覽，確認前不會影響正式牌路。");
    });
    box.appendChild(button);
  }
}
async function importProposal(){
  if(busy||!proposal||proposal.length<1)return;
  if(history().length>0){msg("目前已有牌路紀錄，為避免污染，不允許覆蓋或混合。若要匯入新截圖，請先按「結束分析」。",true);return;}
  if(!root.__BGS_FINAL56__||!root.__BGS256_TEST__){msg("BBB 預測核心尚未就緒，請稍候。",true);return;}
  busy=true;el("bbbScanConfirm").disabled=true;
  const seq=proposal.slice();
  try{
    for(let i=0;i<seq.length;i++){
      const btn=el({B:"btnB",P:"btnP",T:"btnT"}[seq[i]]);
      if(!btn)throw Error("找不到原版 BBB 回報按鈕");
      btn.click();
      if(i%8===7)await new Promise(resolve=>setTimeout(resolve,0));
    }
    const current=history();
    if(current.join("")!==seq.join(""))throw Error("匯入後牌路校驗不一致");
    const storage=root.__BBB_LICENSE_STORAGE__;
    for(let i=0;i<70&&storage?.status().pending_writes>0;i++)await new Promise(r=>setTimeout(r,40));
    if(storage&&(storage.status().pending_writes>0||storage.status().error))throw Error("資料寫入尚未完成或失敗");
    proposal=null;el("bbbScanResult").hidden=true;
    el("bbbScanFile").value="";
    el("btnStart").click();
    msg("已將 "+seq.length+" 局匯入本會員獨立牌路，並自動分析下一局。");
  }catch(error){msg("匯入中斷："+String(error.message||error)+"。請核對歷史紀錄，避免重複匯入。",true);}
  finally{busy=false;el("bbbScanConfirm").disabled=!proposal;}
}
function pointValue(id){
  const v=el(id)?.value?.trim()??"";
  return /^[0-9]$/.test(v)?+v:null;
}
let autoLock=false;
function settlePoints(){
  if(autoLock||busy)return;
  const p=pointValue("macroPlayerScore"),b=pointValue("macroBankerScore");
  if(p===null||b===null)return;
  autoLock=true;
  try{
    const before=history().length;
    const winner=b>p?"B":p>b?"P":"T";
    el({B:"btnB",P:"btnP",T:"btnT"}[winner]).click();
    if(history().length!==before+1){
      msg("點數回報沒有成功寫入歷史，請檢查原版 BBB 儲存訊息。",true);
      return;
    }
    // Frozen Core + Particle500 uses the new B/P/T history on next analysis.
    el("macroPlayerScore").value="";el("macroBankerScore").value="";
    el("btnStart").click();
    el("bbbLastScore").textContent="已記錄第 "+(before+1)+" 局："+(winner==="B"?"莊":winner==="P"?"閒":"和")+"（閒 "+p+" ／ 莊 "+b+"）。已自動分析下一局。";
    // Mobile users can type the next score without extra taps.
  }finally{autoLock=false;}
}
function pointerXY(event){
  const c=el("bbbScanCanvas"),r=c.getBoundingClientRect();
  return [clamp((event.clientX-r.left)/r.width,0,1),clamp((event.clientY-r.top)/r.height,0,1)];
}
function init(){
  const file=el("bbbScanFile");if(!file)return;
  file.addEventListener("change",async()=>{
    const f=file.files?.[0];if(!f)return;
    const job=++scanNo;selection=null;proposal=null;el("bbbScanConfirm").disabled=true;
    el("bbbScanResult").hidden=true;msg("正在讀取截圖…");
    try{const c=await readBitmap(f);if(job!==scanNo)return;sourceCanvas=c;showCanvas();
      // Let mobile browsers paint image before parsing.
      await new Promise(r=>setTimeout(r,0));scan();
    }catch(error){msg(String(error.message||error),true);}
  });
  el("bbbScanProfile").addEventListener("change",()=>{selection=null;showCanvas();scan();});
  el("bbbScanRescan").addEventListener("click",scan);
  el("bbbScanConfirm").addEventListener("click",importProposal);
  const canvas=el("bbbScanCanvas");
  canvas.addEventListener("pointerdown",e=>{if(!sourceCanvas)return;drag=pointerXY(e);canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener("pointermove",e=>{
    if(!drag)return;const at=pointerXY(e);
    selection=[Math.min(at[0],drag[0]),Math.min(at[1],drag[1]),
      Math.abs(at[0]-drag[0]),Math.abs(at[1]-drag[1])];
    showCanvas();
  });
  canvas.addEventListener("pointerup",e=>{
    if(!drag)return;drag=null;
    if(!selection||selection[2]<.06||selection[3]<.018){selection=null;showCanvas();return;}
    showCanvas();scan();
  });
  canvas.addEventListener("pointercancel",()=>{drag=null;});
  for(const id of ["macroPlayerScore","macroBankerScore"]){
    el(id).addEventListener("input",settlePoints);
    el(id).addEventListener("change",settlePoints);
  }
  el("btnEnd")?.addEventListener("click",()=>{
    selection=null;proposal=null;sourceCanvas=null;el("bbbScanFile").value="";
    el("bbbScanVisual").hidden=true;el("bbbScanResult").hidden=true;
    el("macroPlayerScore").value="";el("macroBankerScore").value="";
    el("bbbLastScore").textContent="";
    msg("新牌靴可重新上傳截圖。");
  });
}
root.__BBB_ROAD_SCANNER__={scanImage:scan,history,source:"BGS road_detector ROI/color/reconstruction browser adaptation",maxSide:MAX_SIDE};
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",init,{once:true});else init();
})(window);

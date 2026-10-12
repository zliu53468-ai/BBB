/* BBB desktop screenshot entry — opt-in getDisplayMedia, one frame only.
 * Operates entirely in-memory and feeds the existing BGS-inspired BBB
 * browser road scanner. No capture/recording API used on mobile.
 */
(function(root){
"use strict";
const el=id=>document.getElementById(id),CAPTURE_MAX_SIDE=4096;
let image=null,selection=null,drag=null,working=false,returnFocus=null;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const notify=(text,warning=false)=>{
 const e=el("bbbScanMessage");
 if(e){e.textContent=text;e.classList.toggle("warning",warning);}
};
function supported(){
 return !!(root.isSecureContext&&root.navigator?.mediaDevices?.getDisplayMedia&&
     root.__BBB_ROAD_SCANNER__?.loadCapturedCanvas);
}
function reveal(){
 const btn=el("bbbCaptureScreen");
 if(btn)btn.hidden=!supported();
}
function showOverlay(c){
 image=c;selection=null;drag=null;
 const panel=el("bbbCaptureDialog"),use=el("bbbCaptureUse");
 if(!panel||!use)throw Error("截圖介面未就緒");
 returnFocus=document.activeElement;
 panel.hidden=false;document.documentElement.classList.add("bbb-capturing");
 use.disabled=true;
 paint();
 el("bbbCaptureHint").textContent="按住滑鼠框選六列大路區域，或使用整張截圖";
 el("bbbCaptureCancel")?.focus();
}
function closeOverlay(){
 const panel=el("bbbCaptureDialog");if(panel)panel.hidden=true;
 document.documentElement.classList.remove("bbb-capturing");
 image=null;selection=null;drag=null;
 returnFocus?.focus?.();returnFocus=null;
}
function paint(){
 if(!image)return;
 const c=el("bbbCaptureCanvas"),ctx=c?.getContext("2d");
 if(!c||!ctx)throw Error("無法顯示截圖");
 if(c.width!==image.width)c.width=image.width;
 if(c.height!==image.height)c.height=image.height;
 ctx.drawImage(image,0,0);
 if(selection){
   const [x,y,w,h]=selection;
   ctx.fillStyle="rgba(30,12,68,.25)";
   ctx.fillRect(x,y,w,h);
   ctx.strokeStyle="#eecaff";ctx.lineWidth=Math.max(3,image.width/700);
   ctx.setLineDash?.([12,7]);ctx.strokeRect(x,y,w,h);ctx.setLineDash?.([]);
 }
}
function pixelPoint(event){
 const r=el("bbbCaptureCanvas").getBoundingClientRect();
 if(r.width<=0||r.height<=0)throw Error("截圖畫面尺寸異常");
 return [clamp((event.clientX-r.left)/r.width,0,1)*image.width,
   clamp((event.clientY-r.top)/r.height,0,1)*image.height];
}
function crop(canvas,rect){
 if(!canvas||!rect)throw Error("沒有選擇大路截圖");
 const [rawX,rawY,rawW,rawH]=rect;
 const x=clamp(Math.floor(rawX),0,canvas.width-1),y=clamp(Math.floor(rawY),0,canvas.height-1);
 const w=clamp(Math.round(rawW),0,canvas.width-x),h=clamp(Math.round(rawH),0,canvas.height-y);
 if(w<60||h<30)throw Error("框選區域太小，請框住完整的六列大路");
 const out=document.createElement("canvas");
 out.width=w;out.height=h;
 out.getContext("2d").drawImage(canvas,x,y,w,h,0,0,w,h);
 return out;
}
function useRegion(full=false){
 if(!image)return;
 let captured;
 try{
   captured=full?image:crop(image,selection);
   // Reuse OCR preview/confirm; the scanner will not record history until
   // the user explicitly confirms its B/P/T result.
   closeOverlay();
   root.__BBB_ROAD_SCANNER__.loadCapturedCanvas(captured);
   notify(full?"已擷取整張畫面；請核對掃描預覽。":"已擷取所選大路區域；請核對掃描預覽。");
 }catch(e){el("bbbCaptureHint").textContent=e?.message||"截圖區域有誤";}
}
async function frameFromStream(stream){
 const video=document.createElement("video");
 video.muted=true;video.playsInline=true;video.autoplay=true;
 video.srcObject=stream;
 try{
   const ready=new Promise((resolve,reject)=>{
     if(video.readyState>=2)return resolve();
     let timeout;
     const ok=()=>{clearTimeout(timeout);cleanup();resolve();};
     const fail=()=>{clearTimeout(timeout);cleanup();reject(Error("未能取得螢幕畫面"))};
     const cleanup=()=>{video.removeEventListener("loadeddata",ok);video.removeEventListener("error",fail);};
     video.addEventListener("loadeddata",ok,{once:true});
     video.addEventListener("error",fail,{once:true});
     timeout=setTimeout(fail,6000);
   });
   await video.play();
   await ready;
   if(video.requestVideoFrameCallback){
     await new Promise(resolve=>{
       let done=false;
       const finish=()=>{if(!done){done=true;resolve();}};
       video.requestVideoFrameCallback(finish);
       setTimeout(finish,650);
     });
   }
   const w=video.videoWidth,h=video.videoHeight;
   if(!w||!h)throw Error("截圖沒有有效影像尺寸");
   const scale=Math.min(1,CAPTURE_MAX_SIDE/Math.max(w,h));
   const c=document.createElement("canvas");
   c.width=Math.round(w*scale);c.height=Math.round(h*scale);
   const ctx=c.getContext("2d",{willReadFrequently:true});
   if(!ctx)throw Error("無法讀取擷取影像");
   ctx.drawImage(video,0,0,c.width,c.height);
   return c;
 }finally{
   video.pause?.();video.srcObject=null;
 }
}
async function captureScreen(){
 if(working)return;
 if(!supported()){
   notify("此瀏覽器無法直接截取桌面畫面；請用 Chrome／Edge 的 HTTPS 網頁，或改用原本的上傳圖片。",true);
   return;
 }
 working=true;const btn=el("bbbCaptureScreen");btn.disabled=true;
 let stream=null;
 notify("請在瀏覽器視窗選擇遊戲畫面、視窗或螢幕並按分享；我們只擷取一張截圖。");
 try{
   // Browser permission is mandatory, and can only be requested on user click.
   stream=await root.navigator.mediaDevices.getDisplayMedia({video:true,audio:false});
   const captured=await frameFromStream(stream);
   showOverlay(captured);
   notify("電腦畫面已擷取，分享已停止；請用滑鼠框選大路。");
 }catch(error){
   if(["NotAllowedError","AbortError"].includes(error?.name)){
     notify("已取消螢幕截圖，原本牌路與手機上傳功能不受影響。");
   }else{
     notify("截圖未成功："+String(error?.message||error||"未知錯誤")+"。仍可使用原本的相簿／圖片上傳。",true);
   }
 }finally{
   // Terminate every track immediately, including on crop / decoding errors.
   if(stream)for(const track of stream.getTracks())try{track.stop();}catch(_){}
   btn.disabled=false;working=false;
 }
}
function onKeydown(e){if(!el("bbbCaptureDialog")?.hidden&&e.key==="Escape"){e.preventDefault();closeOverlay();}}
function init(){
 const button=el("bbbCaptureScreen");
 if(!button)return;
 reveal();
 button.addEventListener("click",captureScreen);
 el("bbbCaptureCancel")?.addEventListener("click",closeOverlay);
 el("bbbCaptureUse")?.addEventListener("click",()=>useRegion(false));
 el("bbbCaptureFull")?.addEventListener("click",()=>useRegion(true));
 document.addEventListener("keydown",onKeydown);
 const c=el("bbbCaptureCanvas");
 c?.addEventListener("pointerdown",event=>{
   if(!image||event.button>0)return;
   drag=pixelPoint(event);selection=null;
   c.setPointerCapture?.(event.pointerId);
 });
 c?.addEventListener("pointermove",event=>{
   if(!image||!drag)return;
   const p=pixelPoint(event);
   selection=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),
     Math.abs(drag[0]-p[0]),Math.abs(drag[1]-p[1])];
   paint();
 });
 c?.addEventListener("pointerup",event=>{
   if(!image||!drag)return;
   const p=pixelPoint(event);
   selection=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),
     Math.abs(drag[0]-p[0]),Math.abs(drag[1]-p[1])];drag=null;
   const valid=selection[2]>=60&&selection[3]>=30;
   if(!valid)selection=null;
   el("bbbCaptureUse").disabled=!valid;
   el("bbbCaptureHint").textContent=valid?"已框選大路區域，點「截取框選並掃描」即可。":"請重新拖曳框選完整的六列大路。";
   paint();
 });
 c?.addEventListener("pointercancel",()=>{drag=null;selection=null;el("bbbCaptureUse").disabled=true;paint();});
 el("btnEnd")?.addEventListener("click",()=>{if(image)closeOverlay();});
}
root.__BBB_DESKTOP_CAPTURE__={captureScreen,isSupported:supported,crop};
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",init,{once:true});else init();
})(window);

#!/usr/bin/env node
// BBB desktop capture regression: original phone gallery path stays untouched.
// Tests user-triggered screen permission, one video frame, stopped tracks,
// region selection and handoff to the existing image-scanner API.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
const source=fs.readFileSync("bbb-desktop-capture.js","utf8");
const listeners=new Map(),nodes=new Map(),accepted=[];
let screenRequests=0,stopCount=0,scanned=0;
const get=id=>{
 if(!nodes.has(id)){
  const callbacks=new Map();
  const c={
    id,hidden:id==="bbbCaptureScreen"||id==="bbbCaptureDialog",
    disabled:false,textContent:"",value:"",clientWidth:1000,clientHeight:540,width:1920,height:1080,
    classList:{toggle(){},add(){},remove(){}},focus(){},
    addEventListener(n,cb){callbacks.set(n,cb);},
    setPointerCapture(){},dispatch(n,p){return callbacks.get(n)?.(p);},
    getBoundingClientRect(){return {left:0,top:0,width:1000,height:550};},
    getContext(){return {drawImage(...args){if(args.length)scanned++;},
      fillRect(){},strokeRect(){},setLineDash(){}};}
  };
  nodes.set(id,c);
 }
 return nodes.get(id);
};
const document={
 readyState:"loading",activeElement:{focus(){}},
 documentElement:{classList:{add(){},remove(){}}},
 addEventListener(k,cb){listeners.set(k,cb);},
 createElement(tag){
  if(tag==="video"){
    return {muted:false,playsInline:false,autoplay:false,srcObject:null,
      readyState:3,videoWidth:1920,videoHeight:1080,
      play:async()=>{},pause(){},requestVideoFrameCallback(cb){cb();}};
  }
  if(tag==="canvas")return {width:0,height:0,getContext:()=>({drawImage(){scanned++;}})};
  throw Error("Unexpected tag "+tag);
 },
 getElementById:get
};
const track={stop(){stopCount++;}};
const stream={getTracks:()=>[track]};
const window={isSecureContext:true,navigator:{mediaDevices:{
 getDisplayMedia:async(opts)=>{
  assert.equal(opts.audio,false);
  assert.equal(opts.video,true);
  screenRequests++;return stream;
 }
}},__BBB_ROAD_SCANNER__:{
 loadCapturedCanvas(canvas){assert(canvas.width>0&&canvas.height>0);accepted.push([canvas.width,canvas.height]);}
}};
vm.runInNewContext(source,{window,document,setTimeout,clearTimeout,Promise,Math,Number,
 Error,String,console},{filename:"bbb-desktop-capture.js"});
listeners.get("DOMContentLoaded")();
const btn=get("bbbCaptureScreen");
assert.equal(btn.hidden,false,"supported desktop should show snip icon");
assert.equal(get("bbbScanFile").value,"","mobile/gallery file input starts unchanged");
assert.equal(window.__BBB_DESKTOP_CAPTURE__.isSupported(),true);
await btn.dispatch("click");
assert.equal(screenRequests,1,"one explicit getDisplayMedia permission");
assert.equal(stopCount,1,"screen sharing must stop immediately after still frame");
assert.equal(get("bbbCaptureDialog").hidden,false,"region selector is visible");
assert.equal(get("bbbCaptureUse").disabled,true,"cannot crop before user selects region");
get("bbbCaptureCanvas").dispatch("pointerdown",{button:0,pointerId:1,clientX:200,clientY:160});
get("bbbCaptureCanvas").dispatch("pointermove",{pointerId:1,clientX:700,clientY:430});
get("bbbCaptureCanvas").dispatch("pointerup",{pointerId:1,clientX:700,clientY:430});
assert.equal(get("bbbCaptureUse").disabled,false,"valid drag enables one-shot crop");
get("bbbCaptureUse").dispatch("click");
assert.equal(accepted.length,1,"sends capture into existing BBB image scanner");
assert.equal(accepted[0][0],960,"crop width uses original image resolution");
assert(accepted[0][1]>=525&&accepted[0][1]<=532,"crop height maps displayed canvas into source pixels");
assert.equal(get("bbbCaptureDialog").hidden,true,"overlay closed after selecting screenshot");
assert.equal(get("bbbScanFile").value,"","desktop capture never touches mobile gallery input");
assert.throws(()=>window.__BBB_DESKTOP_CAPTURE__.crop(
 {width:100,height:100},[5,5,6,8]),/太小/,"tiny selection rejected");
await window.__BBB_DESKTOP_CAPTURE__.captureScreen();
get("bbbCaptureFull").dispatch("click");
assert.equal(accepted.length,2,"full screenshot fallback reuses OCR flow");
assert.deepEqual(accepted.at(-1),[1920,1080]);
assert.equal(stopCount,2,"second permission request stopped as well");
window.navigator.mediaDevices.getDisplayMedia=async()=>{
 const e=new Error("Permission denied");e.name="NotAllowedError";throw e;
};
await window.__BBB_DESKTOP_CAPTURE__.captureScreen();
assert.match(get("bbbScanMessage").textContent,/已取消/, "cancellation is harmless");
assert.equal(accepted.length,2,"cancel must not write another screenshot");
console.log(JSON.stringify({ok:true,checks:["desktop feature detect","user permission","single video frame","tracks stopped","mouse crop maps native pixels","OCR handoff","full screenshot fallback","permission cancel safe","mobile gallery untouched"]}));

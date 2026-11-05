'use strict';
// Local Chromium CDP test; never touches the user's normal browser profile.
const fs=require('fs'),assert=require('assert/strict');
(async()=>{
 const tabs=await(await fetch('http://127.0.0.1:9333/json/list')).json();
 const ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
 let n=0;const pending=new Map();ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(pending.has(m.id)){const {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(JSON.stringify(m.error))):resolve(m.result);}});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++n;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await send('Page.enable');
 const results={browser:await send('Browser.getVersion'),environment:'headless Chrome, local fixture; mobile CPU 4x throttle is simulation, not physical phone'};
 for(const mobile of [false,true]){
  await send('Emulation.setCPUThrottlingRate',{rate:mobile?4:1});
  await send('Emulation.setDeviceMetricsOverride',{width:mobile?390:1280,height:mobile?844:900,deviceScaleFactor:1,mobile});
  await send('Emulation.setTouchEmulationEnabled',{enabled:mobile,maxTouchPoints:mobile?5:1});
  await send('Page.navigate',{url:'http://127.0.0.1:8765/tools/v41_browser_fixture.html?mobile='+mobile});
  for(let i=0;i<100;i++){if(await evaluate('Boolean(window.__BBB_PHYSICAL_V4_PANEL__)'))break;await new Promise(r=>setTimeout(r,50));}
  const ui=await evaluate(`(()=>{localStorage.clear();const p=document.getElementById('pointPlayerFinal'),b=document.getElementById('pointBankerFinal');p.value='5';b.value='3';p.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));return {coarse:matchMedia('(pointer: coarse)').matches,afterEnter:Boolean(__BBB_PHYSICAL_V4_PANEL__.latestReport)};})()`);
  assert.equal(ui.coarse,mobile);assert.equal(ui.afterEnter,false);
  if(mobile)await evaluate(`document.getElementById('btnPointAnalyze').click()`);
  const responsiveness=await evaluate(`(async()=>{let ticks=0;const timer=setInterval(()=>ticks++,10),start=performance.now();while(!__BBB_PHYSICAL_V4_PANEL__.latestReport&&performance.now()-start<15000)await new Promise(r=>setTimeout(r,20));clearInterval(timer);return {ticks,complete:Boolean(__BBB_PHYSICAL_V4_PANEL__.latestReport),batchCount:__BBB_PHYSICAL_V4_PANEL__.latestReport?.sampling_stability?.batches};})()`);
  assert(responsiveness.complete);assert(responsiveness.ticks>0,'UI event loop blocked during calculation');assert([3,5].includes(responsiveness.batchCount));
  const replayCheck=await evaluate(`(async()=>{const ui=__BBB_PHYSICAL_V4_PANEL__,seed=ui.latestReport.randomization.seed,stored=ui.audit.at(-1).analysis_seed;const wait=async()=>{const start=performance.now();while(!ui.latestReport&&performance.now()-start<15000)await new Promise(r=>setTimeout(r,20));if(!ui.latestReport)throw Error('worker response missing');};document.getElementById('pointPlayerFinal').value='2';document.getElementById('pointBankerFinal').value='8';document.getElementById('btnPointAnalyze').click();await wait();document.getElementById('btnBack').click();await wait();return {stored:stored===seed,replayed:ui.latestReport.randomization.seed===seed,mode:ui.latestReport.randomization.mode};})()`);
  assert.equal(replayCheck.stored,true);assert.equal(replayCheck.replayed,true);assert.equal(replayCheck.mode,'explicit_seed_replay');
  const bench=await evaluate(`(()=>{const {PhysicalPointV4,CONFIG}=BBBPhysicalV4;const versions={A:{...CONFIG,simulationsPerParticle:160,eorParticleCount:16,noiseGuardMultiplier:.7},B:CONFIG};const result={};for(const [name,c]of Object.entries(versions)){const m=name==='B'?new BBBStability.StablePhysicalPointV4(c):new PhysicalPointV4(c),times=[];for(const pair of [[0,0],[5,3],[2,8],[9,9],[8,1],[4,6],[1,4],[7,2]]){const t=performance.now();m.analyze(...pair,{seed:0});times.push(performance.now()-t);}const sorted=times.slice().sort((a,b)=>a-b);result[name]={times,mean:times.reduce((a,b)=>a+b,0)/times.length,p95:sorted[sorted.length-1],max:sorted[sorted.length-1]};}return result;})()`);
  results[mobile?'mobile_4x_cpu':'desktop']={input:ui,benchmark_ms:bench,worker_responsiveness:responsiveness};
  assert(bench.B.max<15000,'Background analysis exceeds timeout budget');
 }
 await send('Emulation.setCPUThrottlingRate',{rate:1});
 fs.writeFileSync('validation/v41_browser_performance.json',JSON.stringify(results,null,2));
 console.log(JSON.stringify(results,null,2));send('Browser.close').catch(()=>{});ws.close();
})().catch(e=>{console.error(e);process.exitCode=1;});

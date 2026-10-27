'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync('auth-client.js','utf8');
const functions=source.slice(source.indexOf('async function boundedEndSync'),source.indexOf('async function unlock'));
async function scenario(kind){
 const controls=[{disabled:false},{disabled:true}],events=[],values=new Map();
 const panel={querySelectorAll:()=>controls,setAttribute:()=>events.push('busy'),removeAttribute:()=>events.push('unbusy')};
 const msg={textContent:'',classList:{add(){}}};
 const storage={end:async()=>{events.push('deleted');return kind!=='delete_fail';},acknowledgeRemoteEnd:async()=>events.push('ack')};
 const ctx={URL,Date,Promise,Array,Error,document:{getElementById:()=>msg},panel,sessionStorage:{setItem:(k,v)=>values.set(k,v)},window:{__BBB_LICENSE_STORAGE__:storage,location:{href:'https://example.test/BBB/?foo=1#panel',replace:url=>events.push(url)}},signedChallenge:async()=>{events.push('remote');if(kind==='hang')return new Promise(()=>{});if(kind==='offline')throw Error('offline');return {ok:true,generation:2};},setTimeout:(fn,ms)=>setTimeout(fn,Math.min(ms,5)),clearTimeout};
 vm.createContext(ctx);vm.runInContext('let authorized=true,endBusy=false,busy=false,activeCode="code",activeLicenseId="id";\n'+functions,ctx);
 await ctx.onEndAnalysis();
 const navigations=events.filter(x=>x.startsWith('https:'));
 if(kind==='delete_fail'){assert.equal(navigations.length,0);assert.equal(controls[0].disabled,false);assert.equal(controls[1].disabled,true);assert(!events.includes('remote'));}
 else {assert.equal(navigations.length,1);const u=new URL(navigations[0]);assert(u.searchParams.has('analysis_reset'));assert.equal(u.searchParams.get('foo'),'1');assert.equal(u.hash,'#panel');assert(events.indexOf('deleted')<events.indexOf(navigations[0]));assert(controls.every(x=>x.disabled));if(kind==='hang'||kind==='offline')assert.equal(values.size,1);await ctx.onEndAnalysis();assert.equal(events.filter(x=>x.startsWith('https:')).length,1);}
}
(async()=>{for(const kind of ['ok','offline','hang','delete_fail'])await scenario(kind);console.log('End Analysis: local purge before fresh navigation, bounded remote sync, failure recovery and duplicate-click protection passed');})().catch(e=>{console.error(e);process.exitCode=1;});

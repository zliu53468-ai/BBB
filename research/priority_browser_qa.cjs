'use strict';
// Local isolated presentation fixture only; no production authentication bypass.
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.BBB_PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.BBB_CHROME_PATH?{executablePath:process.env.BBB_CHROME_PATH}:{})}),results=[];
 for(const width of [320,390,1365]){
  const page=await browser.newPage({viewport:{width,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',r=>r.request().url().startsWith('http://127.0.0.1:8898/')?r.continue():r.abort());
  await page.route('**/auth-client.js*',r=>r.fulfill({contentType:'application/javascript',body:'/* isolated local QA: no auth code or remote calls */'}));
  await page.goto('http://127.0.0.1:8898/');await page.evaluate(()=>{document.getElementById('licenseGate').hidden=true;document.getElementById('predictionApp').hidden=false;});
  for(const f of ['app256forward.js','app256continuation.js','particle_filter_runtime.js','macro_ema.js','browser_calibration.js','point_conditioned_pf.js','physics_priority_runtime.js','final_probability_runtime.js'])await page.addScriptTag({url:'http://127.0.0.1:8898/'+f});
  await page.evaluate(()=>window.__BGS_FINAL56__.loadModels());
  for(const side of 'BPTBPPBBPT')await page.locator('#btn'+side).click();
  await page.locator('#priorityPointInputs summary').click();await page.locator('#macroPlayerScore').fill('2');await page.locator('#macroBankerScore').fill('6');await page.locator('#btnB').click();
  const value=await page.evaluate(()=>{
   const api=window.__BGS_FINAL56__,history=JSON.parse(localStorage.getItem('bgs256d_frozen_6x15_forward_v18')).history,p=api.applyFinalPrediction(history);api.renderPrediction(p,history.length);
   return {history_length:history.length,point_weight:p.physics_priority.point_weight,status:document.getElementById('physicsPriorityStatus').textContent,overflow:document.documentElement.scrollWidth>innerWidth};
  });assert.equal(value.point_weight,.65);assert(!value.overflow);assert.match(value.status,/信心/);assert.equal(errors.length,0);
  await page.screenshot({path:path.resolve('research/reports/priority_v1_smoke/ui-'+width+'.png'),fullPage:true});results.push({width,...value,errors});await page.close();
 }
 await browser.close();fs.writeFileSync('research/reports/priority_v1_smoke/browser-qa.json',JSON.stringify({passed:true,scope:'local isolated UI fixture; real auth unchanged and separately regression tested',results},null,2));console.log(JSON.stringify(results));
})().catch(e=>{console.error(e);process.exitCode=1;});

'use strict';
// Synthetic validation, not a real-table backtest. Freeze settings before holdout.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {performance}=require('node:perf_hooks');
const root=path.resolve(__dirname,'..');
const {PhysicsForecastFirstEngine}=require(root+'/physics_forecast_first_runtime.js');
const guard=require(root+'/point_signal_quality_guard.js');
const deplete=require(root+'/deplete_mc_runtime.js');
const reference=require(root+'/validation/deplete_exact_reference.json');
const cases=1000000,seed=20261105;
const table=[];let ms=0,maxMs=0;
for(let player=0;player<10;player++)for(let banker=0;banker<10;banker++){
 const model=new PhysicsForecastFirstEngine({independentFinalScoreMode:true,exactSingleScore:true});
 const old=model.syncHistory([banker>player?'B':player>banker?'P':'T'],[{player,banker}]);
 const start=performance.now(),fused=deplete.attach(old,{player,banker},{mode:'candidate'});
 const duration=performance.now()-start;ms+=duration;maxMs=Math.max(maxMs,duration);
 table.push({player,banker,primary:guard.decide(old).direction,candidate:guard.decide(fused).direction,
  primaryRaw:old.ev_banker>=old.ev_player?'B':'P',
  primary_ev_banker:old.ev_banker,primary_ev_player:old.ev_player,
  candidate_ev_banker:fused.ev_banker,candidate_ev_player:fused.ev_player,
  deplete_ev_banker:fused.deplete_mc.ev_banker,deplete_ev_player:fused.deplete_mc.ev_player,
  completed_shoes:fused.deplete_mc.completed_shoes,samples:fused.deplete_mc.samples});
}
// Separate target simulator: physical cards and partial Fisher-Yates. This does
// not call the PF, conditional proposal, paired-control deal or exact reference.
let state=seed>>>0;
function integer(bound){
 const limit=4294967296-(4294967296%bound);let v;
 do{state^=state<<13;state^=state>>>17;state^=state<<5;v=state>>>0;}while(v>=limit);
 return v%bound;
}
const original=new Uint8Array(416);let n=128;
for(let r=1;r<10;r++)for(let j=0;j<32;j++)original[n++]=r;
const deck=new Uint8Array(416);let position=0;
function draw(){const j=position+integer(416-position),v=deck[j];deck[j]=deck[position];deck[position++]=v;return v;}
function deal(){
 const a=draw(),b=draw(),c=draw(),d=draw();let p=(a+c)%10,q=(b+d)%10;
 if(p<8&&q<8){
  if(p<=5){const third=draw();p=(p+third)%10;
   const take=q<=2||q===3&&third!==8||q===4&&third>=2&&third<=7||
    q===5&&third>=4&&third<=7||q===6&&(third===6||third===7);
   if(take)q=(q+draw())%10;
  }else if(q<=5)q=(q+draw())%10;
 }
 return {player:p,banker:q,outcome:q>p?'B':p>q?'P':'T'};
}
const names=['primary','candidate','always_banker','primary_on_candidate_entries','banker_on_candidate_entries'];
const stats=Object.fromEntries(names.map(k=>[k,{bets:0,wins:0,losses:0,ties:0,profit:0,profit2:0}]));
const paired={primary:{sum:0,sum2:0},banker:{sum:0,sum2:0}};
const bins=Array.from({length:100},()=>[0,0,0]);
const trace=new Uint8Array(cases*2);
function settle(name,choice,outcome){
 if(choice==='SKIP')return 0;const s=stats[name];s.bets++;let gain=0;
 if(outcome==='T')s.ties++;
 else if(choice===outcome){s.wins++;gain=choice==='B'?.95:1;}
 else{s.losses++;gain=-1;}
 s.profit+=gain;s.profit2+=gain*gain;return gain;
}
for(let i=0;i<cases;i++){
 deck.set(original);position=0;const current=deal(),k=current.player*10+current.banker;
 const prediction=table[k]; // Freeze before dealing target.
 const result=deal().outcome;trace[2*i]=k;trace[2*i+1]='BPT'.indexOf(result);bins[k][trace[2*i+1]]++;
 settle('primary',prediction.primary,result);
 const c=settle('candidate',prediction.candidate,result);
 settle('always_banker','B',result);
 const oldMatched=settle('primary_on_candidate_entries',prediction.candidate==='SKIP'?'SKIP':prediction.primaryRaw,result);
 const bMatched=settle('banker_on_candidate_entries',prediction.candidate==='SKIP'?'SKIP':'B',result);
 for(const [key,value] of [['primary',c-oldMatched],['banker',c-bMatched]]){paired[key].sum+=value;paired[key].sum2+=value*value;}
}
function wilson(w,n){if(!n)return null;const z=1.96,d=1+z*z/n,c=(w/n+z*z/(2*n))/d;
 const h=z*Math.sqrt(w/n*(1-w/n)/n+z*z/(4*n*n))/d;return[c-h,c+h];}
for(const s of Object.values(stats)){
 s.coverage=s.bets/cases;s.hit_rate_excluding_ties=s.wins/(s.wins+s.losses);
 s.hit_rate_ci95=wilson(s.wins,s.wins+s.losses);s.roi=s.profit/s.bets;
 const se=Math.sqrt((s.profit2/s.bets-s.roi*s.roi)/(s.bets-1));s.roi_ci95=[s.roi-1.96*se,s.roi+1.96*se];
}
for(const p of Object.values(paired)){const mean=p.sum/cases,se=Math.sqrt((p.sum2/cases-mean*mean)/(cases-1));p.mean_profit_difference_per_case=mean;p.ci95=[mean-1.96*se,mean+1.96*se];}
const count=key=>table.reduce((a,r)=>(a[r[key]]++,a),{B:0,P:0,SKIP:0});
const error=prefix=>Math.sqrt(table.reduce((a,r,i)=>{const p=reference.rows[i],b=.95*p[0]-p[1],v=p[1]-p[0];
 return a+(r[prefix+'_ev_banker']-b)**2+(r[prefix+'_ev_player']-v)**2;},0)/200);
let maxZ=0;
for(let k=0;k<100;k++){const n=bins[k].reduce((a,b)=>a+b);for(let j=0;j<3;j++){
 const p=reference.rows[k][j];maxZ=Math.max(maxZ,Math.abs(bins[k][j]/n-p)/Math.sqrt(p*(1-p)/n));}}
if(maxZ>6)throw Error('independent validation simulator disagrees with reference');
const promotion=paired.primary.ci95[0]>0&&paired.banker.ci95[0]>0&&stats.candidate.roi_ci95[0]>0;
const report={version:'BBB_DEPLETE_VALIDATION_V1',kind:'synthetic_independent_two_hand_holdout',
 scenarios:cases,dealt_hands:2*cases,seed,trained_parameters:0,config:deplete.CONFIG,
 primary_display_counts:count('primary'),candidate_display_counts:count('candidate'),
 ev_rmse:{primary:error('primary'),deplete:error('deplete'),candidate:error('candidate')},
 mean_auxiliary_ms:ms/100,max_auxiliary_ms:maxMs,stats,paired,
 max_conditional_bin_standardized_error:maxZ,promotion_eligible:promotion,
 recommended_mode:promotion?'candidate':'shadow',
 trace_sha256:crypto.createHash('sha256').update(trace).digest('hex'),
 limitations:['Synthetic data do not establish live-casino gains.','PF and deplete share the same observation and prior; not independent evidence.','Monte Carlo precision gains do not guarantee hit-rate gains.'],table};
fs.writeFileSync(root+'/validation/deplete_validation.json',JSON.stringify(report,null,2)+'\n');
if(process.env.WRITE_DEPLETE_TRACE==='1')fs.writeFileSync(root+'/validation/deplete_cases.bin',trace);
console.log(JSON.stringify({scenarios:cases,primary:report.primary_display_counts,candidate:report.candidate_display_counts,
 ev_rmse:report.ev_rmse,primary_hit:stats.primary.hit_rate_excluding_ties,candidate_hit:stats.candidate.hit_rate_excluding_ties,
 primary_roi:stats.primary.roi,candidate_roi:stats.candidate.roi,paired,promotion_eligible:promotion,mean_ms:ms/100,maxZ},null,2));

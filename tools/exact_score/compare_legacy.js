'use strict';
const fs=require('node:fs');
const {PhysicsForecastFirstEngine}=require('../../physics_forecast_first_runtime.js');
const guard=require('../../point_signal_quality_guard.js');
const exact=require('../../exact_score_runtime.js');
const rows=['player\tbanker\told_display\texact_display\told_evB\told_evP'];
for(let p=0;p<10;p++)for(let b=0;b<10;b++){
 const m=new PhysicsForecastFirstEngine({independentFinalScoreMode:true,exactSingleScore:true});
 const r=m.syncHistory([b>p?'B':p>b?'P':'T'],[{player:p,banker:b}]);
 const e=exact.evaluate(p,b);
 rows.push([p,b,guard.decide(r).direction,guard.decide(e).direction,r.ev_banker,r.ev_player].join('\t'));
}
fs.writeFileSync(__dirname+'/comparison.tsv',rows.join('\n')+'\n');

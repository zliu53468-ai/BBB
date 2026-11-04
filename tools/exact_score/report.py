from pathlib import Path
import csv,json,math,hashlib,statistics
ROOT=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
rows=list(csv.DictReader((D/'exact_scores.tsv').open(),delimiter='\t'))
comparison=list(csv.DictReader((D/'comparison.tsv').open(),delimiter='\t'))
lines=(D/'million_results.tsv').read_text().splitlines();cut=lines.index('BIN\tB\tP\tT')
metrics=list(csv.DictReader(lines[:cut],delimiter='\t'))
bins=[[int(x) for x in line.split('\t')] for line in lines[cut+1:]]
def wilson(w,n):
 if n==0:return None
 z=1.959963984540054;c=(w/n+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(w/n*(1-w/n)/n+z*z/(4*n*n))/(1+z*z/n)
 return [c-h,c+h]
policies={}
for m in metrics:
 n=int(m['bets']);w=int(m['wins']);loss=int(m['losses']);tie=int(m['ties']);profit=float(m['profit']);p2=float(m['profit2'])
 if m['policy'].startswith('paired_'):
  mean=profit/n;se=math.sqrt(max(0,p2/n-mean*mean)/(n-1))
  paired={'comparison':'exact B candidate vs legacy on identical legacy entry cases; zero outside entries','profit_difference_per_scenario':mean,'ci95':[mean-1.96*se,mean+1.96*se]};continue
 mean=profit/n if n else None;se=math.sqrt(max(0,p2/n-mean*mean)/(n-1)) if n>1 else None
 policies[m['policy']]={'bets':n,'wins':w,'losses':loss,'ties':tie,'coverage':n/1e6,'non_tie_hit_rate':w/(w+loss) if w+loss else None,'hit_rate_ci95':wilson(w,w+loss),'profit_units':profit,'roi_per_bet':mean,'roi_ci95':[mean-1.96*se,mean+1.96*se] if se is not None else None}
max_z=0;chi=0;calibration=[]
for k,cb,cp,ct in bins:
 r=rows[k];n=cb+cp+ct;ps=[float(r[o]) for o in ['B','P','T']];cs=[cb,cp,ct]
 zs=[abs(c/n-p)/math.sqrt(p*(1-p)/n) for c,p in zip(cs,ps)]
 max_z=max(max_z,*zs);chi+=sum((c-n*p)**2/(n*p) for c,p in zip(cs,ps))
 calibration.append({'player':k//10,'banker':k%10,'cases':n,'observed':dict(zip(['B','P','T'],[c/n for c in cs])),'exact':dict(zip(['B','P','T'],ps)),'max_standardized_error':max(zs)})
assert sum(sum(b[1:]) for b in bins)==1000000
assert max_z<6,'Independent sampler disagrees with exact table'
assert abs(sum(float(r['evidence']) for r in rows)-1)<1e-10
errors=[abs(float(c[o])-v) for c,r in zip(comparison,rows) for o,v in [('old_evB',.95*float(r['B'])-float(r['P'])),('old_evP',float(r['P'])-float(r['B']))]]
trace=D/'million_results.tsv.cases.bin'
assert trace.stat().st_size==2000000
report={'version':'BBB_EXACT_SINGLE_SCORE_MILLION_V1','validation_type':'synthetic_independent_two_hand_scenarios_not_live_casino_backtest','independent_scenarios':1000000,'dealt_hands':2000000,'seed':20261104,'generator':'MillionScoreValidation.cs physical deck partial Fisher-Yates; decisions frozen before target deal','training_samples':0,'assumption':'fresh 8-deck shoe before each first hand; no live shoe composition/depth observed','exact_legal_sequences':339400,'exact_merged_depletions':6958,'all_input_pairs':100,'probability_mass_error':abs(sum(float(r['evidence']) for r in rows)-1),'max_bin_standardized_error':max_z,'conditional_multinomial_chi_square':chi,'conditional_multinomial_df':200,'per_score_calibration':calibration,'policies':policies,'paired_comparison':paired,'legacy_ev_absolute_error_vs_exact':{'mean':statistics.mean(errors),'max':max(errors)},'all_exact_ev_negative':all(max(.95*float(r['B'])-float(r['P']),float(r['P'])-float(r['B']))<0 for r in rows),'exact_v4_display_counts':{o:sum(c['exact_display']==o for c in comparison) for o in ['B','P','SKIP']},'raw_trace_schema':'2 bytes per independent case: player*10+banker, then next outcome 0=B 1=P 2=T','raw_trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'table_sha256':hashlib.sha256((ROOT/'exact_score_table.js').read_bytes()).hexdigest(),'limitations':['Synthetic cases do not establish real-world prediction advantage.','One million cases validate sampling/calibration, not a guaranteed hit-rate improvement.','Candidate direction is separate from actionable V4 display: exact V4 abstains on all 100 score pairs.','Unknown live shoe depth, discarded cards and rule variants are not observed or reconstructed.']}
(ROOT/'validation/exact_score_million_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:report[k] for k in ['independent_scenarios','max_bin_standardized_error','conditional_multinomial_chi_square','exact_v4_display_counts','legacy_ev_absolute_error_vs_exact']},indent=2))

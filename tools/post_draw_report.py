"""Post-draw EV and 4/5/6-card calibration report. Standard library only."""
import argparse,json,math
from pathlib import Path
SOURCES=('direct_physics','pf_posterior','mixed')
STAGES={'all':lambda r:True,'1-20':lambda r:1<=r<=20,'21-45':lambda r:21<=r<=45,'46-70':lambda r:46<=r<=70,'<=40':lambda r:r<=40,'41-50':lambda r:41<=r<=50,'>50':lambda r:r>50,'50-70':lambda r:50<=r<=70}
def settled_ev(probabilities,banker_commission=.05,player_commission=0.):
    values=[probabilities[k] for k in ('B','P','T')]
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values) or abs(sum(values)-1)>1e-6:raise ValueError('Invalid post-draw probabilities')
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in (banker_commission,player_commission)):raise ValueError('Invalid commission')
    B,P,T=[v/sum(values) for v in values]
    return {'banker':(1-banker_commission)*B-P,'player':(1-player_commission)*P-B}
def build_report(payload):
    rows=payload.get('rows',[]) if isinstance(payload,dict) else payload
    if not isinstance(rows,list):raise ValueError('rows must be a list')
    stages={}
    for stage,predicate in STAGES.items():
        selected=[r for r in rows if predicate(int(r['round_index']))];sources={}
        for source in SOURCES:
            forecasts=[];matched=[];invalid_distributions=0
            for row in selected:
                forecast=(row.get('draw_report') or {}).get(source) or {}
                if not forecast.get('available'):continue
                if forecast.get('normalized') is False:invalid_distributions+=1;continue
                p=[forecast['probabilities'][str(k)+'_cards'] for k in (4,5,6)]
                if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in p) or abs(sum(p)-1)>1e-6:raise ValueError('Invalid draw count distribution')
                forecasts.append(p)
                actual=(row.get('observed_hand') or {}).get('card_count')
                # Never infer physical card counts from B/P/T or from a model estimate.
                if type(actual) is int and actual in (4,5,6):matched.append((p,actual))
            n=len(forecasts);m=len(matched)
            mean=lambda items:[sum(p[k] for p in items)/len(items) for k in range(3)] if items else None
            sources[source]={'forecast_rows':n,'invalid_or_renormalized_rows':invalid_distributions,'forecast_mean':mean(forecasts),'measured_card_count_rows':m,
                'matched_forecast_mean':mean([p for p,_ in matched]),
                'actual_proportions':[sum(actual==k for _,actual in matched)/m for k in (4,5,6)] if m else None,
                'multiclass_brier':sum(sum((p[i]-int(actual==k))**2 for i,k in enumerate((4,5,6))) for p,actual in matched)/m if m else None,
                'calibration_status':'measured_diagnostic_only' if m else 'not_evaluable_without_measured_card_count'}
        stakes=profit=non_tie=wins=0;gapmax=0.;audits=0
        for row in selected:
            audit=row.get('settlement_audit') or {};final=audit.get('final');policy=audit.get('policy')
            if not final or not policy:continue
            audits+=1
            expected=settled_ev(final['probabilities'],policy['banker_commission'],policy['player_commission'])
            gapmax=max(gapmax,abs(final['banker']-expected['banker']),abs(final['player']-expected['player']))
            direction=row.get('direction',row.get('bet_direction'));outcome=row.get('outcome',row.get('actual_outcome'))
            if direction not in ('B','P') or outcome not in ('B','P','T'):continue
            stakes+=1
            if outcome=='T':continue
            non_tie+=1
            if outcome==direction:wins+=1;profit+=1-policy['banker_commission' if direction=='B' else 'player_commission']
            else:profit-=1
        stages[stage]={'rows':len(selected),'draw_counts':sources,'settlement_audit_rows':audits,'maximum_ev_formula_error':gapmax if audits else None,'ev_formula_consistent':gapmax<=1e-12 if audits else None,'placed_bets':stakes,'hit_rate_on_bets':wins/non_tie if non_tie else None,'realized_ev_per_bet':profit/stakes if stakes else None}
    return {'schema_version':1,'data_kind':payload.get('data_kind','unspecified') if isinstance(payload,dict) else 'unspecified','card_count_order':[4,5,6],'probability_basis':'completed_hand_after_third_card_rules; unconditional B/P/T for EV','stages':stages,'warning':'Calibration diagnostics require observed physical card_count. B/P/T alone cannot validate draw-count calibration. Synthetic data is mechanics evidence only.','production_promotion_allowed':False}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('output',type=Path);a=parser.parse_args()
    payload=json.loads(a.input.read_text(encoding='utf-8'));result=build_report(payload)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'rows':result['stages']['all']['rows'],'measured_rows':result['stages']['all']['draw_counts']['mixed']['measured_card_count_rows'],'ev_consistent':result['stages']['all']['ev_formula_consistent']}))

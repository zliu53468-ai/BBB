import ast,unittest
from pathlib import Path
from tools.post_draw_report import settled_ev,build_report
class PostDrawReportTests(unittest.TestCase):
 def test_payoffs_and_bad_units(self):
  self.assertAlmostEqual(settled_ev({'B':.5,'P':.4,'T':.1})['banker'],.075)
  self.assertEqual(settled_ev({'B':0,'P':0,'T':1})['player'],0)
  with self.assertRaises(ValueError):settled_ev({'B':.51,'P':.49,'T':.1})
 def test_unmeasured_count_is_not_inferred(self):
  d={'available':True,'probabilities':{'4_cards':.5,'5_cards':.3,'6_cards':.2}}
  r=build_report({'data_kind':'fixture','rows':[{'round_index':50,'outcome':'B','draw_report':{'mixed':d}}]})
  s=r['stages']['50-70']['draw_counts']['mixed']
  self.assertEqual(s['forecast_rows'],1);self.assertIsNone(s['actual_proportions']);self.assertIsNone(s['multiclass_brier'])
 def test_measured_count_and_full_tie_push(self):
  d={'available':True,'probabilities':{'4_cards':.5,'5_cards':.3,'6_cards':.2}}
  ev={'probabilities':{'B':.5,'P':.4,'T':.1},**settled_ev({'B':.5,'P':.4,'T':.1})}
  rows=[{'round_index':r,'direction':'B','outcome':o,'draw_report':{'mixed':d},'observed_hand':{'card_count':n},'settlement_audit':{'policy':{'banker_commission':.05,'player_commission':0},'final':ev}} for r,o,n in [(49,'B',4),(50,'T',5),(51,'P',6)]]
  s=build_report({'rows':rows})['stages']['all']
  self.assertAlmostEqual(s['realized_ev_per_bet'],-.05/3)
  self.assertTrue(s['ev_formula_consistent'])
  self.assertEqual(s['draw_counts']['mixed']['measured_card_count_rows'],3)
  self.assertEqual(s['draw_counts']['mixed']['actual_proportions'],[1/3]*3)
 def test_python_physics_banker_rules(self):
  tree=ast.parse(Path('physics_feature_extractor.py').read_text(encoding='utf-8'))
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_banker_draws');ns={}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_physics_rule','exec'),ns)
  for b in range(10):
   for p in [None,*range(10)]:
    expected=b<=5 if p is None else True if b<=2 else p!=8 if b==3 else 2<=p<=7 if b==4 else 4<=p<=7 if b==5 else p in (6,7) if b==6 else False
    self.assertEqual(ns['_banker_draws'](b,p),expected)
if __name__=='__main__':unittest.main()

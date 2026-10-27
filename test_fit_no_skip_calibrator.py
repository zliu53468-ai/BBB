import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.fit_no_skip_calibrator import fit

class CalibrationFitTests(unittest.TestCase):
    def test_bounded_identity_and_no_fake_gain(self):
        rows=[{'outcome':'B' if i%2 else 'P','probe':{'raw':{'directional':{'B':.5}}}} for i in range(200)]
        model=fit(rows,.005)
        self.assertAlmostEqual(model['slope'],1,places=5)
        self.assertAlmostEqual(model['intercept'],0,places=5)
        self.assertTrue(model['optimizer_success'])
        with self.assertRaisesRegex(ValueError,'100'):fit(rows[:10],.005)

    def test_unused_holdout_labels_cannot_change_fit(self):
        rows=[{'shoe_id':str(s),'outcome':'B' if r%2 else 'P','probe':{'raw':{'directional':{'B':.5}}}} for s in range(15) for r in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache.json';manifest=root/'report.json'
            manifest.write_text(json.dumps({'data_kind':'TEST_FIXTURE','input_sha256':'fixture','seed':1}),encoding='utf-8')
            results=[]
            for version in range(2):
                if version:
                    for row in rows:
                        if int(row['shoe_id'])>=12:row['outcome']='P'
                cache.write_text(json.dumps({'rows':rows,'hashes':{'runtime':'fixture'}}),encoding='utf-8')
                output=root/f'candidate{version}.json'
                subprocess.run([sys.executable,'tools/fit_no_skip_calibrator.py','--cache',str(cache),'--training-report',str(manifest),'--output',str(output)],check=True,capture_output=True)
                result=json.loads(output.read_text(encoding='utf-8'));results.append(result['candidates'])
                self.assertEqual(result['train_shoes'],9)
                self.assertFalse(result['production_promotion_allowed'])
            self.assertEqual(results[0],results[1])

if __name__=='__main__':unittest.main()

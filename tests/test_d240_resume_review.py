import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from d240_resume_job import verify_reuse,verify_four
from d234_support import read
class ResumeReviewTests(unittest.TestCase):
 def test_saved_baseline_is_reusable(self):
  m=read(ROOT/'reports/d240_resume_review/reuse_manifest.json')
  rows=verify_reuse(ROOT/'reports/d238_two_task/evidence',ROOT,m)
  self.assertEqual(len(rows),2);self.assertEqual({x['seed'] for x in rows},{7})
 def test_reuse_rejects_changed_seed_or_file(self):
  original=read(ROOT/'reports/d240_resume_review/reuse_manifest.json')
  for mutation in ['seed','file']:
   m=copy.deepcopy(original)
   if mutation=='seed':m['seed']=8
   else:m['files'][next(iter(m['files']))]='0'*64
   with self.assertRaises(ValueError):verify_reuse(ROOT/'reports/d238_two_task/evidence',ROOT,m)
 def test_comparison_requires_four_and_invariants(self):
  r=dict(step=4,updates=[dict(step=i) for i in range(1,5)],kl_policy='full_batch_mean_v2',frozen_adapters_unchanged=True,reload_logps_exact=True)
  verify_four(r)
  for key,value in [('step',3),('reload_logps_exact',False),('frozen_adapters_unchanged',False),('kl_policy','legacy')]:
   with self.assertRaises(ValueError):verify_four(r|{key:value})
if __name__=='__main__':unittest.main()

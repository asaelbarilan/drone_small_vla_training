"""Exercise orchestration review boundary with bounded fake external stages."""
import sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
import d237_job

class FourUpdateReview(unittest.TestCase):
 def run_job(self,reported_step):
  with tempfile.TemporaryDirectory() as td:
   code=Path(td)/'code';root=Path(td)/'run'
   p=code/'reports/d234_full_ppo_preparation/partitions.json';p.parent.mkdir(parents=True)
   p.write_text(json.dumps(dict(training=[dict(source='train'+str(i),kind=str(i)) for i in range(8)],development=[dict(source='dev')],validation=[dict(source='val')])))
   calls=[]
   def external(stages,*args,**kwargs):
    s=stages[0];calls.append(s)
    out=Path(s['required_outputs'][0]);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(step=reported_step,frozen_adapters_unchanged=True,reload_logps_exact=True)))
   with patch.object(d237_job,'CODE',code),patch.object(d237_job,'materialize'),patch.object(d237_job,'score_controls',return_value={}),patch.object(d237_job,'admission',return_value=40),patch.object(d237_job,'verified_checkpoint',return_value=root/'checkpoint'),patch.object(d237_job,'comparison',return_value={'rows':[]}),patch('d234_support.behavior_gate',return_value={'continue_training':True}),patch.object(d237_job,'run_stages',side_effect=external),patch.object(sys,'argv',['job','--root',str(root),'--wall-seconds','21600']):
    if reported_step==4:d237_job.main()
    else:
     with self.assertRaises(ValueError):d237_job.main()
   if reported_step==4:
    self.assertFalse(json.loads((root/'completion.json').read_text())['automatic_continuation'])
    self.assertTrue((root/'review_required.json').exists())
    self.assertEqual(calls[-1]['name'],'after_dev')
   else:self.assertFalse(any(s['name']=='after_dev' for s in calls))
   train=[s for s in calls if 'training_to_' in s['name']]
   self.assertEqual(len(train),1)
   argv=train[0]['argv'];self.assertEqual(argv[argv.index('--pause-after-updates')+1],'4')
   self.assertFalse(any('validation' in s['name'] for s in calls))
 def test_gain_cannot_continue_automatically(self):self.run_job(4)
 def test_short_update_count_cannot_be_called_four(self):self.run_job(3)
if __name__=='__main__':unittest.main()

import unittest,sys,tempfile,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'scripts/rl')]
from numeric_goals import instruction_goal_mode
from d234_support import materialize,score_controls,comparison,admission,behavior_gate
class PreparationTest(unittest.TestCase):
 def test_instruction_routing(self):
  self.assertEqual(instruction_goal_mode('Rotate 60 degrees to the right',automatic=True),'numeric')
  self.assertEqual(instruction_goal_mode('Fly around the tree',automatic=True),'generated')
  self.assertEqual(instruction_goal_mode('Rotate 60 degrees to the right'),'generated')
  with self.assertRaises(ValueError):instruction_goal_mode('Fly around the tree',strict=True)
 def test_actual_task_integrity_and_reward_controls(self):
  q=json.loads((R/'reports/d234_full_ppo_preparation/partitions.json').read_text())
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'tasks';materialize(R,q['training']+q['development'],p)
   report=score_controls(p);self.assertEqual(len(report['rows']),95)
   broken=dict(q['training'][0],sha256='0'*64)
   with self.assertRaises(ValueError):materialize(R,[broken],Path(d)/'bad')
 def test_matched_actual_flights_and_count_rejection(self):
  e=R/'reports/d232_aws_ppo/evidence'
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'comparison.json'
   # D232 attempts contain seed8 too, so make an explicit matching baseline subset.
   b=Path(d)/'before/episodes';b.mkdir(parents=True)
   for f in (e/'attempts/episodes').glob('*_r0.json'):(b/f.name).write_bytes(f.read_bytes())
   result=comparison(b.parent,e/'after',out,4);self.assertEqual(len(result['rows']),4)
   decision=behavior_gate(b.parent,e/'after',result)
   self.assertFalse(decision['continue_training']);self.assertEqual(decision['changed_action_tasks'],[])
   with self.assertRaises(ValueError):comparison(b.parent,e/'after',out,8)
 def test_behavior_gain_regression_and_goal_confound(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d)/'before';a=Path(d)/'after'
   for p in (b,a):(p/'episodes').mkdir(parents=True)
   old=dict(task='x',seed=7,samples=[dict(ids=[1],progress_text='Left +01.0,+00.0,+00.0,+000')])
   new=dict(task='x',seed=7,samples=[dict(ids=[2],progress_text='Left +01.0,+00.0,+00.0,+000')])
   (b/'episodes/sim_x_r0.json').write_text(json.dumps(old));(a/'episodes/sim_x_r0.json').write_text(json.dumps(new))
   row=dict(before=dict(geometric_complete=False,reward=0.,endpoint_position_error_m=1.),after=dict(geometric_complete=True,reward=1.,endpoint_position_error_m=.1))
   self.assertTrue(behavior_gate(b,a,dict(rows=[row]))['continue_training'])
   new['samples'][0]['progress_text']='Left +01.0,+00.0,+00.0,+360'
   (a/'episodes/sim_x_r0.json').write_text(json.dumps(new))
   self.assertFalse(behavior_gate(b,a,dict(rows=[row]))['continue_training'])
   new['samples'][0]['progress_text']=old['samples'][0]['progress_text'];(a/'episodes/sim_x_r0.json').write_text(json.dumps(new))
   row['before']['geometric_complete']=True;row['after']['geometric_complete']=False
   self.assertFalse(behavior_gate(b,a,dict(rows=[row]))['continue_training'])
 def test_validation_samples_cannot_be_admitted(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'episodes').mkdir();(p/'episodes/usage.json').write_text('{"training_eligible":false}')
   with self.assertRaisesRegex(ValueError,'validation-only'):admission(p,p,p/'absent.json',p,1)
if __name__=='__main__':unittest.main()

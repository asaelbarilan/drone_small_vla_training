import unittest,sys,math,json,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from numeric_goals import numeric_goal,numeric_progress,numeric_action_instruction
class NumericGoalsTest(unittest.TestCase):
 def test_all_explicit_practice_goals(self):
  paths=list((ROOT/'reports/d221_vertical_landing/tasks').glob('*.json'))+list((ROOT/'reports/d224_reward_gated_ppo/tasks').glob('*.json'))
  self.assertGreaterEqual(len(paths),6)
  for p in paths:
   t=json.loads(p.read_text());g=numeric_goal(t['instruction']);ref=t['reference_path_preprocessed'][-1]
   with self.subTest(task=p.name):
    self.assertLess(math.dist(g[:3],[v/100 for v in ref[:3]]),.02);self.assertAlmostEqual(g[3],ref[4])
 def test_numpy_http_state_is_json_serializable(self):
  import numpy as np
  for yaw in [np.float64(0),np.float64(60)]:
   used,meta=numeric_progress('Rotate 60 degrees to the right',np.zeros(3),yaw)
   self.assertIs(type(meta['goal_arrived']),bool)
   json.dumps(dict(used=used,meta=meta),allow_nan=False)
 def test_axis_and_sign(self):
  self.assertEqual(numeric_goal('Rotate 60 degrees to the left'),[0,0,0,-60])
  self.assertEqual(numeric_goal('Ascend vertically by 4 metres from the starting height, then stop.'),[0,0,4,0])
 def test_ambiguous_rejected(self):
  for s in ['Land 4 meters','Descend to 4 metres','Ascend 9 meters at 85 degrees','Rotate 60 degrees to the right then land','Rotate nan degrees to the right','Rotate 0 degrees to the right','Descend vertically by 0 metres from the starting height, then stop.']:
   with self.subTest(s=s),self.assertRaises(ValueError):numeric_goal(s)
 def test_early_zero_regression(self):
  cmd='Descend vertically by 4 metres from the starting height, then stop.'
  left,meta=numeric_progress(cmd,[.0648,.0032,-3.172],-.043)
  self.assertAlmostEqual(left[2],-.828);self.assertFalse(meta['goal_arrived'])
 def test_overshoot_and_boundary(self):
  cmd='Descend vertically by 4 metres from the starting height, then stop.'
  self.assertGreater(numeric_progress(cmd,[0,0,-4.3],0)[0][2],0)
  self.assertFalse(numeric_progress(cmd,[0,0,-3.899],0)[1]['goal_arrived'])
  self.assertEqual(numeric_progress(cmd,[0,0,-3.901],0)[0],[0]*4)
  self.assertFalse(numeric_progress(cmd,[0,0,-4],1.01)[1]['goal_arrived'])
 def test_start_frame_and_winding(self):
  cmd='Travel forward 9 metres along a straight 85-degree upward slope, then stop. The distance is along the slope, not a target altitude.'
  left,_=numeric_progress(cmd,[0,0,0],90)
  self.assertAlmostEqual(left[0],0);self.assertLess(left[1],0);self.assertGreater(left[2],8.9)
  self.assertEqual(numeric_progress('Rotate 360 degrees to the right',[0,0,0],359.5)[0],[0]*4)
  self.assertEqual(numeric_progress('Rotate 360 degrees to the right',[0,0,0],0)[0][3],360)
 def test_canonical_slope_no_reference(self):
  command='Travel forward 5 metres along a straight 75-degree downward slope, then stop. The distance is along the slope, not a target altitude.'
  self.assertEqual(numeric_action_instruction(command),'Move 1.294 metres forward and 4.830 metres down simultaneously, then stop.')
  self.assertEqual(numeric_action_instruction('Rotate 60 degrees to the right'),'Rotate 60 degrees to the right')
 def test_small_turn_and_submetre_rejection(self):
  self.assertFalse(numeric_progress('Rotate 5 degrees to the right',[0,0,0],4.4)[1]['goal_arrived'])
  self.assertTrue(numeric_progress('Rotate 5 degrees to the right',[0,0,0],4.6)[1]['goal_arrived'])
  with self.assertRaises(ValueError):numeric_goal('Descend vertically by 0.5 metres from the starting height, then stop.')
 def test_nonfinite_state(self):
  with self.assertRaises(ValueError):numeric_progress('Rotate 60 degrees to the right',[float('nan'),0,0],0)
 def test_server_numeric_branch_bypasses_generated_goal(self):
  # Run actual nested server function with a model encoder that must never be called.
  tree=ast.parse((ROOT/'scripts/uav_flow_eval_server.py').read_text())
  loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qwen_loader')
  nodes=[n for n in loader.body if isinstance(n,ast.FunctionDef) and n.name in ('progress_line','numeric_for')]
  env=dict(numeric_goals=True,auto_numeric_goals=False,STATE={'yaw_total':0},progress_text=lambda v:tuple(v))
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_server_branch','exec'),env)
  text,info=env['progress_line']({},None,[0,0,-3.172,0],'Descend vertically by 4 metres from the starting height, then stop.')
  self.assertAlmostEqual(text[2],-.828);self.assertIsNone(info['progress_parsed'])
 def test_actual_server_action_prompt_preserves_original_box_prompt(self):
  import numpy as np
  tree=ast.parse((ROOT/'scripts/uav_flow_eval_server.py').read_text())
  loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qwen_loader')
  nodes=[n for n in loader.body if isinstance(n,ast.FunctionDef) and n.name in ('predict','progress_line','numeric_for')]
  observed={}
  def box_line(row,image):
   observed['box_prompt']=row['prompt'];return '{}',{}
  def encode(processor,tokenizer,row,chunk,**kwargs):
   observed['action_prompt']=row['prompt'];return {'input_ids':np.array([[1,2]])}
  env=dict(numeric_goals=True,auto_numeric_goals=False,STATE={'yaw_total':0},progress_text=lambda v:str(v),
           OFFICIAL_PROMPT='State {state}; {instruction}',composite=None,box=True,progress=True,
           box_line=box_line,encode=encode,processor=None,tokenizer=None,chunk=8,precision='bf16')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_server_prediction','exec'),env)
  command='Travel forward 5 metres along a straight 75-degree downward slope, then stop. The distance is along the slope, not a target altitude.'
  result=env['predict'](None,[0,0,0,0],command,context_only=True)
  self.assertIn(command,observed['box_prompt'])
  self.assertIn('1.294 metres forward and 4.830 metres down',observed['action_prompt'])
  self.assertEqual(result['prompt'],observed['action_prompt'])
  with self.assertRaises(ValueError):env['predict'](None,[0,0,0,0],'Land on the roof',context_only=True)

if __name__=='__main__':unittest.main()

"""D229 offline preference regression; no model, simulator or network."""
import sys,json,unittest,importlib.util,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from movement_reward import movement_reward
from audit_movement_reward import line,controls,variants
spec=importlib.util.spec_from_file_location('old_reward',ROOT/'reports/d229_arrival_reward/movement_reward_v3.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
class ArrivalRewardTests(unittest.TestCase):
    def test_recorded_pair_and_clean(self):
        task=ROOT/'reports/d224_reward_gated_ppo/tasks/sim_2025-03-27_19-11-33.json'
        ref=json.loads(task.read_text())['reference_path_preprocessed']; rows=[]
        for i in (0,1):
            p=ROOT/f'reports/d228_local_numeric_flights/finish/episodes/sim_2025-03-27_19-11-33_r{i}.json'
            ep=json.loads(p.read_text());a=old.movement_reward(ep['poses'],ref,kind='Ascend/Descend',stopped=True)
            self.assertAlmostEqual(a['reward'],ep['reward']['reward'],places=9)
            b=movement_reward(ep['poses'],ref,kind='Ascend/Descend',stopped=True)
            self.assertAlmostEqual(sum(b['transition_rewards']),b['reward'],places=12)
            self.assertEqual(a['complete'],b['complete'])
            rows.append(dict(seed=7+i,source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),old=a['reward'],new=b['reward'],endpoint_m=b['endpoint_position_error_m'],terminal_credit=b['terminal_credit'],complete=b['complete']))
        clean=movement_reward(ref,ref,kind='Ascend/Descend',stopped=True)
        self.assertGreater(clean['reward'],rows[0]['new']);self.assertGreater(rows[0]['new'],rows[1]['new'])
        (ROOT/'reports/d231_overshoot_ppo/comparison.json').write_text(json.dumps(dict(recorded=rows,synthetic_clean=clean['reward'],ordering='clean > recorded detour arrival > recorded near-inaction',training=False),indent=2))
    def test_controls_and_no_loop_profit(self):
        for z in (-5,5):
            ref=line(x=1.3,z=z);paths=variants(ref)
            scores={k:movement_reward(v,ref,kind='Ascend/Descend',stopped=True) for k,v in paths.items()}
            for k in ('stationary','early_stop','wrong_direction','overshoot','off_path_detour','reversal'):
                self.assertGreater(scores['correct']['reward'],scores[k]['reward'],k)
            self.assertGreater(scores['off_path_detour']['reward'],scores['stationary']['reward'])
            loop=np.vstack([ref,ref[-2::-1]])
            self.assertLessEqual(movement_reward(loop,ref,kind='Ascend/Descend',stopped=True)['reward'],0)
            detour=paths['off_path_detour']
            dup=movement_reward(np.repeat(detour,3,axis=0),ref,kind='Ascend/Descend',stopped=True)
            self.assertAlmostEqual(dup['reward'],scores['off_path_detour']['reward'],places=8)
            dense=np.vstack([p+(q-p)*t for p,q in zip(detour,detour[1:]) for t in (0,.5)]+[detour[-1]])
            self.assertAlmostEqual(movement_reward(dense,ref,kind='Ascend/Descend',stopped=True)['reward'],dup['reward'],places=3)
            for poses,events in [(detour,{'collision':True}),([ref[0],ref[-1]],{})]:
                result=movement_reward(poses,ref,kind='Ascend/Descend',stopped=True,events=events)
                self.assertLessEqual(result['reward'],-1+1e-9);self.assertFalse(result['complete'])
            self.assertEqual(movement_reward(detour,ref,kind='Ascend/Descend',stopped=False)['terminal_credit'],0)
    def test_overshoot_loses_progress_and_recovery_restores_it(self):
        for sign in (-1,1):
            ref=line(z=9*sign)
            score=lambda a:movement_reward(a,ref,kind='Ascend/Descend',stopped=False)['reward']
            idle=score(line());clean=score(ref)
            self.assertLess(score(line(z=18*sign)),idle)
            self.assertLess(score(line(z=20*sign)),idle)
            self.assertLess(score(line(z=10*sign)),clean)
            self.assertLess(score(line(z=-9*sign)),idle)
            recover=np.vstack([line(z=18*sign),np.linspace(line(z=18*sign)[-1],ref[-1],101)[1:]])
            self.assertGreater(score(recover),idle)
            self.assertLess(score(recover),clean)

    def test_other_kinds_unchanged(self):
        for kind,ref in controls().items():
            if kind=='Ascend/Descend':continue
            for poses in variants(ref).values():
                kw=dict(kind=kind,stopped=True,events={'landed':True})
                a=old.movement_reward(poses,ref,**kw);b=movement_reward(poses,ref,**kw)
                self.assertEqual(a['reward'],b['reward']);self.assertEqual(a['complete'],b['complete'])
if __name__=='__main__':unittest.main()


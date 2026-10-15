import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from critic_observation import critic_observation
from audit_movement_reward import line
from audit_accepted_goal_stopping import KinematicHarness
from flight_collector import collect_episode

class ObservationTests(unittest.TestCase):
    def observe(self,p,r,**kw):
        return critic_observation(p,r,executed_moves=kw.pop('executed_moves',len(p)-1),
                                  kind='Ascend/Descend',max_moves=128,**kw)['features']
    def test_no_future_pose_or_outcome_leakage(self):
        r=line(x=3,z=2,n=20);p=r[:10].copy();a=self.observe(p,r)
        future=np.vstack([p,np.full((5,6),np.nan)])
        self.assertEqual(a,self.observe(future,r,executed_moves=9))
        future[10:]=1e9
        self.assertEqual(a,self.observe(future,r,executed_moves=9))
    def test_same_current_pose_different_route_history(self):
        r=line(x=3,z=2,n=20);p=r.copy();p[:,1]=120*np.sin(np.linspace(0,np.pi,len(p)))
        p[[0,-1]]=r[[0,-1]]
        clean=self.observe(r,r);detour=self.observe(p,r)
        self.assertEqual(clean['endpoint_error_m'],detour['endpoint_error_m'])
        self.assertEqual(clean['heading_error_deg'],detour['heading_error_deg'])
        self.assertEqual(clean['accepted_now'],1.);self.assertEqual(detour['accepted_now'],0.)
        self.assertGreater(detour['max_route_deviation'],.75)
    def test_quiet_count_matches_execution_termination(self):
        r=line(x=3,z=2,n=20);p=np.vstack([r[:10],np.repeat(r[9:10],8,axis=0)])
        state=self.observe(p,r);self.assertEqual(state['quiet_moves'],8.)
        env=KinematicHarness(p[0]);commands=list(p[1:])+[p[-1]]*4;cursor=0
        def policy(obs):
            nonlocal cursor
            part=commands[cursor:cursor+8];cursor+=len(part)
            return dict(proprio=obs['pose'][:3]+[obs['pose'][4]],
                        poses=[list(a[:3])+[float(np.deg2rad(a[4]))] for a in part])
        e=collect_episode(env,policy,lambda o:{},r,max_steps=128,reward_kind='Ascend/Descend',reward_stopping_mode='accepted_goal')
        self.assertEqual(sum(e['executed_counts']),len(p)-1+2)
        with self.assertRaises(ValueError):self.observe(e['poses'],r)
    def test_current_goal_requires_position_and_heading(self):
        r=line(x=3,z=2,n=20)
        self.assertEqual(self.observe(r[:10],r)['accepted_now'],0.)
        p=r.copy();p[:,4]=np.linspace(0,6,len(p))
        self.assertEqual(self.observe(p,r)['accepted_now'],0.)
        self.assertEqual(self.observe(r,r)['accepted_now'],1.)
    def test_start_duplicate_does_not_invent_quiet_history(self):
        r=line(x=3,z=2,n=20);s=self.observe(r[:1],r)
        self.assertEqual(s['quiet_moves'],0.);self.assertEqual(s['remaining_moves'],128.)
    def test_records_departure_from_prior_acceptance(self):
        r=line(x=3,z=2,n=20);p=np.vstack([r,r[-1]+[-60,0,0,0,0,0]])
        s=self.observe(p,r);self.assertEqual(s['accepted_now'],0.);self.assertEqual(s['accepted_previously'],1.)
        self.assertGreater(s['motion_cost_since_acceptance'],0.)
    def test_input_unchanged_and_invalid_boundary_rejected(self):
        r=line(x=3,z=2,n=20);old=r.copy();self.observe(r,r);np.testing.assert_array_equal(r,old)
        for n in [-1,len(r),128]:
            with self.assertRaises(ValueError):self.observe(r,r,executed_moves=n)
        with self.assertRaises(ValueError):self.observe(r,r,collision_before='unknown')
    def test_does_not_expose_future_reward_value_or_action(self):
        r=line(x=3,z=2,n=20);s=self.observe(r[:5],r)
        self.assertFalse(set(s)&{'mc_target','return','value','gae','action','ids','terminal_outcome'})
        self.assertEqual(s['collision_known'],0.)
        self.assertEqual(self.observe(r[:5],r,collision_before=True)['accepted_now'],0.)
if __name__=='__main__':unittest.main()

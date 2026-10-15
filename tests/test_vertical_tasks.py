import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from vertical_tasks import repair_vertical_task,vertical_reward,landing_evidence
from audit_movement_reward import line

class VerticalTaskTests(unittest.TestCase):
    def task(self):
        return dict(instruction='Climb to 9 metres altitude',instruction_unified='Ascend 9.0 meters at 30 degrees',
                    motion_kind='Ascend/Descend',reference_path_preprocessed=line(x=7.7942286,z=4.5).tolist())
    def samples(self):
        return [dict(time_s=t,clearance_m=0,speed_m_s=0,contact=True,collision=False,
                     surface_id='pad',source='validated_simulator_contact') for t in [0,.4,.8,1.2]]
    def test_slope_is_not_vertical_altitude(self):
        task,spec=repair_vertical_task(self.task())
        self.assertAlmostEqual(spec['target_delta_m'][2],4.5)
        self.assertIn('along',task['instruction'])
        self.assertTrue(vertical_reward(task['reference_path_preprocessed'],task,stopped=True)['complete'])
        self.assertFalse(vertical_reward(line(x=7.7942286,z=9),task,stopped=True)['complete'])
    def test_ref_disagreement_rejected(self):
        task=self.task();task['reference_path_preprocessed']=line(z=9).tolist()
        with self.assertRaises(ValueError):repair_vertical_task(task)
    def test_numeric_land_becomes_descent(self):
        t=self.task();t.update(instruction_unified='Land 4.0 meters',motion_kind='Land',
                              reference_path_preprocessed=line(z=-4).tolist())
        new,spec=repair_vertical_task(t)
        self.assertEqual(new['motion_kind'],'Ascend/Descend')
        self.assertFalse(spec['training_approved'])
        self.assertEqual(t['motion_kind'],'Land')
    def test_actual_land_is_blocked(self):
        t=self.task();t.update(instruction_unified='Land beside tree',motion_kind='Land')
        self.assertEqual(repair_vertical_task(t)[1]['status'],'blocked_contact_and_landing_site')
    def test_tamper_guard(self):
        t,_=repair_vertical_task(self.task());t['reference_path_preprocessed'][-1][2]+=1
        with self.assertRaises(ValueError):vertical_reward(line(z=4.5),t,stopped=True)
    def test_contact_positive(self):
        self.assertTrue(landing_evidence(self.samples(),surface_id='pad',now_s=1.2)['landed'])
    def test_contact_negatives(self):
        for field,value in [('contact',False),('collision',True),('clearance_m',.5),
                            ('speed_m_s',1.),('surface_id','wall'),('source','pose_inference')]:
            samples=self.samples();samples[-1][field]=value
            self.assertFalse(landing_evidence(samples,surface_id='pad',now_s=1.2)['landed'],field)
    def test_freshness_dwell_and_duplicates(self):
        self.assertFalse(landing_evidence(self.samples(),surface_id='pad',now_s=2.)['landed'])
        self.assertFalse(landing_evidence(self.samples()[-2:],surface_id='pad',now_s=1.2)['landed'])
        s=self.samples();s[-1]['time_s']=.8
        self.assertFalse(landing_evidence(s,surface_id='pad',now_s=1.2)['landed'])
        s=self.samples();s[1]['contact']=False
        self.assertFalse(landing_evidence(s,surface_id='pad',now_s=1.2)['landed'])
if __name__=='__main__':unittest.main()

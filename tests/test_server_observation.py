import base64,hashlib,io,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import uav_flow_eval_server as server

class ObservationTests(unittest.TestCase):
    def test_bootstrap_generates_context_without_sampling(self):
        image=Image.new('RGB',(4,4));buf=io.BytesIO();image.save(buf,format='PNG')
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def predictor(*args):self.fail('bootstrap sampled an action')
            def context(image,pose,instruction):
                calls.append((pose.tolist(),instruction))
                return dict(prompt='p',box_text='',progress_text='',context_input_ids=[7])
            predictor.observe_context=context
            old=server.STATE.copy()
            try:
                server.STATE.clear();server.STATE.update(predict=predictor,rollout_dir=Path(d),episode=2,
                    call=9,last_yaw=170.,yaw_total=170.)
                response=server.app.test_client().post('/observe',json=dict(image=base64.b64encode(buf.getvalue()).decode(),proprio=[1,2,3,-170],instr='pass left'))
                self.assertEqual(response.status_code,200);r=response.get_json()
                self.assertTrue(r['observation_only']);self.assertEqual(server.STATE['call'],9)
                self.assertEqual(server.STATE['yaw_total'],190.)
                self.assertEqual(r['image_sha256'],hashlib.sha256((Path(d)/r['image']).read_bytes()).hexdigest())
                self.assertEqual(calls,[([1.,2.,3.,-170.],'pass left')])
            finally:server.STATE.clear();server.STATE.update(old)

if __name__=='__main__':unittest.main()

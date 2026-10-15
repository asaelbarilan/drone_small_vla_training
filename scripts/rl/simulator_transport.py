"""Transport adapters for the explicit collector; importing never launches a sim."""
import base64,io,math,time
import numpy as np
from PIL import Image

class UnrealReadback:
    """Wrap an already-reset UnrealCV environment. The caller owns reset/launch.

    Calibrate actor heading after reset instead of assuming its mesh's 180-degree
    offset is a world/camera heading. Camera alignment remains a preflight check.
    """
    def __init__(self, unrealcv, actor, initial_pose, set_camera, *, settle_seconds=.1, initial_command_yaw=None):
        self.api=unrealcv;self.actor=actor;self.origin=np.asarray(initial_pose[:3],float)
        self.theta=math.radians(initial_pose[4]);self.set_camera=set_camera
        self.settle=settle_seconds;self.moves=0;self.rotation_corrections=[]
        location=np.asarray(self.api.get_obj_location(actor),float)
        if np.linalg.norm(location-self.origin)>.05:raise ValueError('reset position readback mismatch')
        self.initial_actor_yaw=float(self.api.get_obj_rotation(actor)[1])
        self.initial_command_yaw=self.initial_actor_yaw if initial_command_yaw is None else float(initial_command_yaw)
    def execute(self,command):
        x,y,z,yaw=map(float,command);c,s=math.cos(self.theta),math.sin(self.theta)
        position=self.origin+np.array([c*x-s*y,s*x+c*y,z])
        self.api.set_obj_location(self.actor,position.tolist())
        command_yaw=self.initial_command_yaw+math.degrees(yaw)
        self.api.set_rotation(self.actor,command_yaw)
        self.moves+=1
        if self.settle:
            time.sleep(self.settle);started=time.monotonic();deadline=started+5.;corrected=False;previous=None;stable=0
            target=self.initial_actor_yaw+math.degrees(yaw)
            while True:
                actual=float(self.api.get_obj_rotation(self.actor)[1])
                stable=stable+1 if previous is not None and abs((actual-previous+180)%360-180)<=.001 else 0
                previous=actual
                error=(target-actual+180)%360-180
                if abs(error)<=.005:break
                # Sub-degree simulator steady-state/rounding residuals are corrected
                # using measured feedback; never weaken the execution tolerance.
                if not corrected and time.monotonic()-started>=1.5 and stable>=5 and abs(error)<=.05:
                    command_yaw+=error;self.api.set_rotation(self.actor,command_yaw);corrected=True
                    self.rotation_corrections.append(dict(move=self.moves,target=target,actual=actual,error=error,command=command_yaw))
                if time.monotonic()>=deadline:
                    raise TimeoutError(f'rotation failed: move={self.moves} target={target} actual={actual} error={error} command={command_yaw}')
                time.sleep(.02)
    def observe(self):
        position=np.asarray(self.api.get_obj_location(self.actor),float)-self.origin
        angle=float(self.api.get_obj_rotation(self.actor)[1])-self.initial_actor_yaw
        c,s=math.cos(self.theta),math.sin(self.theta)
        pose=[c*position[0]+s*position[1],-s*position[0]+c*position[1],position[2],0.,(angle+180)%360-180,0.]
        self.set_camera()
        image=Image.fromarray(self.api.get_image(0,'lit')).convert('RGB')
        return dict(pose=pose,image=image,pose_source='simulator_readback',id=str(self.moves))

class PolicyHTTP:
    def __init__(self,url,instruction,task,passage=0,timeout=120):
        import requests
        self.session=requests.Session();self.url=url.rstrip('/');self.instruction=instruction
        self.task=task;self.passage=passage;self.timeout=timeout
    def _request(self,route,observation):
        buffer=io.BytesIO();observation['image'].resize((224,224)).save(buffer,format='PNG')
        p=observation['pose']
        r=self.session.post(self.url+route,json=dict(image=base64.b64encode(buffer.getvalue()).decode(),
            proprio=[p[0],p[1],p[2],p[4]],instr=self.instruction),timeout=self.timeout)
        r.raise_for_status();return r.json()
    def reset(self,seed=None):
        r=self.session.post(self.url+'/reset',json={} if seed is None else dict(seed=seed),timeout=self.timeout);r.raise_for_status()
        if seed is not None and r.json().get('seed')!=seed:raise ValueError('server did not acknowledge episode seed')
    def __call__(self,observation):
        result=self._request('/predict',observation)
        record=result['record']
        if not record.get('ids') or record.get('poses')!=result['action']:
            raise ValueError('server response lacks authoritative sampled record')
        return record|dict(task=self.task,passage=self.passage)
    def observe_context(self,observation):return self._request('/observe',observation)

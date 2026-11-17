"""Actual MuJoCo replay: overview, top view, and the wrist camera."""
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from ..simulation.tabletop_runtime import TabletopRuntime as Runtime
from ..tasks.tabletop_cartesian import CartesianTabletopConfig as Config


class Recorder:
    def __init__(self):
        self.frames=[];self.renderers=[];self.last_time=-1.
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
        self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:]=[.15,0,1.15];self.camera.distance=3.4;self.camera.azimuth=135;self.camera.elevation=-25
    def tick(self,robot,phase):
        now=float(robot.data.time)
        if now>=self.last_time and now-self.last_time<.199:return
        self.last_time=now
        if not self.renderers:
            robot.model.vis.global_.offwidth=max(robot.model.vis.global_.offwidth,960)
            robot.model.vis.global_.offheight=max(robot.model.vis.global_.offheight,720)
            self.renderers=[mujoco.Renderer(robot.model,height=720,width=960),
                            mujoco.Renderer(robot.model,height=220,width=300),
                            mujoco.Renderer(robot.model,height=220,width=300)]
        cams=[self.camera,'top_view','wrist_rgb'];images=[]
        for renderer,camera in zip(self.renderers,cams):
            renderer.update_scene(robot.data,camera=camera);images.append(Image.fromarray(renderer.render()))
        debug=getattr(robot,'vision_debug',None)
        if debug:
            images[1]=Image.fromarray(debug['mask']).convert('RGB').resize((300,220))
            images[2]=Image.fromarray(debug['frame'].rgb).resize((300,220))
        frame=images[0];frame.paste(images[1],(652,98));frame.paste(images[2],(652,340))
        draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,90),fill=(18,23,28))
        draw.text((10,7),'RealMan EE | traditional startup + Cartesian PPO | simulation 4x',fill='white',font=self.font)
        draw.text((10,34),f'pose={robot.config.observation_pose} | phase={phase} | t={now:.2f}s',fill='white',font=self.font)
        p=robot.tcp_pose()[0];caption=f'TCP=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m'
        if debug:
            q=debug['measurement']['object_position_base_m'];caption+=f' | RGB-D base=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}) m'
        draw.text((10,60),caption,fill='white',font=self.font)
        draw.text((660,104),'HSV binary mask' if debug else 'Top view',fill='white',font=self.font)
        draw.text((660,345),'Policy RGB input' if debug else 'Wrist camera (simulation convention)',fill='white',font=self.font)
        self.frames.append(frame)
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.frames[0].save(path,save_all=True,append_images=self.frames[1:],duration=50,loop=1,optimize=False)
        self.frames[-1].save(path.with_suffix('.png'))
        self.frames[0].save(path.with_name(path.stem+'_start.png'))
    def close(self):
        for renderer in self.renderers:renderer.close()


def run(args):
    output=getattr(args,'output',None)
    if output and any(p.exists() for p in (output,output.with_suffix('.png'))):raise ValueError('Choose a new output path')
    recorder=Recorder() if output else None
    tick=recorder.tick if recorder else None
    try:
        if args.command=='setup':
            cfg=Config(observation_pose=getattr(args,'observation_pose',None) or 'table_down')
            robot=Runtime(cfg);report,rotation=robot.prepare(cfg.object_xy,tick=tick)
            result=dict(startup=report,physics=robot.gripper.statistics(),task_actions=0,
                        desired_tcp_m=list(cfg.observation_tcp),camera_optical_frame_calibrated=False)
            path=args.json
        else:
            from ..rl.ppo_tabletop_cartesian import load,evaluate
            model,config=load(args.model)
            result=evaluate(model,config,[args.seed],full=not args.saved_task,tick=tick,
                            observation_pose=getattr(args,'observation_pose',None),
                            object_radius=getattr(args,'object_radius',None))
            result['model_sha256']=hashlib.sha256(args.model.read_bytes()).hexdigest()
            path=output.with_suffix('.json')
        if path.exists():raise ValueError('Choose a new JSON output path')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
        if recorder:recorder.save(output)
    finally:
        if recorder:recorder.close()
    print(result.get('summary',result.get('startup')),flush=True)
    if args.command=='setup':return 0
    return int(result['summary']['successes']!=1)

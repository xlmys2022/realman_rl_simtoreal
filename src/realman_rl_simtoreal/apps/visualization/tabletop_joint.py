"""回放关节桌面 PPO，生成全景与夹爪近景 GIF、PNG 和评估记录。"""
import os

from ...utils.artifacts import file_sha256, write_json


def run(args):
    """按种子回放保存的课程或完整关节桌面任务。"""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from ...rl.ppo_tabletop import evaluate_model, load_checkpoint

    model,config=load_checkpoint(args.model)
    frames=[];renderers=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[.2,0,1.15];camera.distance=3.4;camera.azimuth=135;camera.elevation=-25
    zoom=mujoco.MjvCamera();mujoco.mjv_defaultCamera(zoom)
    zoom.distance=.3;zoom.azimuth=135;zoom.elevation=-20
    def tick(env,info):
        if env._elapsed_steps!=1 and env._elapsed_steps%4 and not env._episode_done:return
        if not renderers:
            env.model.vis.global_.offwidth=max(960,env.model.vis.global_.offwidth)
            env.model.vis.global_.offheight=max(720,env.model.vis.global_.offheight)
            renderers.extend([mujoco.Renderer(env.model,height=720,width=960),mujoco.Renderer(env.model,height=240,width=320)])
            zoom.lookat[:]=env._initial_object;zoom.lookat[2]+=.025
        renderers[0].update_scene(env.data,camera=camera);renderers[1].update_scene(env.data,camera=zoom)
        frame=Image.fromarray(renderers[0].render());close=Image.fromarray(renderers[1].render())
        frame.paste(close,(632,104));draw=ImageDraw.Draw(frame)
        draw.rectangle((630,102,953,345),outline=(90,190,240),width=2)
        draw.rectangle((0,0,960,95),fill=(18,23,28))
        label='SAVED CURRICULUM' if args.saved_task else 'FULL TABLETOP TASK'
        draw.text((12,8),f'PPO | {label} | simulation, 4x',font=font,fill='white')
        draw.text((12,36),f't={env.data.time:.2f}s | clearance={1000*info["support_clearance_m"]:.2f}mm | hold={info["hold_time_s"]:.3f}s',font=font,fill='white')
        start=','.join(f'{v:g}' for v in info['initial_q_deg'])
        draw.text((12,64),f'initial joints=[{start}] deg | jaw={1000*info["actual_width_m"]:.1f}mm',font=font,fill='white')
        frames.append(frame)
    try:result=evaluate_model(model,config,[args.seed],full=not args.saved_task,tick=tick)
    finally:
        for renderer in renderers:renderer.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=50,loop=1,optimize=False)
    frames[-1].save(args.output.with_suffix('.png'))
    result.update(model=str(args.model.resolve()),model_sha256=file_sha256(args.model),
                  replay_speed=4,frames=len(frames),full_task_override=not args.saved_task)
    write_json(args.output.with_suffix('.json'),result)
    print(result['summary']);print('Saved',args.output)
    return int(result['summary']['successes']!=1)

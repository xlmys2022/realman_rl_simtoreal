"""提供模型参数和优化器指纹、Reach 来源摘要，并转发历史文件操作入口。"""
import hashlib
from ...paths import PROJECT_ROOT
from ...utils.artifacts import append_json, empty_directory, file_hashes, runtime_versions, write_json


def versions():
    # Keep Torch optional for callers using only JSON and provenance helpers.
    import torch
    return runtime_versions(torch_cuda=torch.version.cuda)


def asset_hashes():
    files = [p for p in (PROJECT_ROOT/'assets/robots/rml63').rglob('*') if p.is_file()]
    files += [PROJECT_ROOT/'src/realman_rl_simtoreal'/name for name in
              ('envs/reach/reach_env.py', 'envs/reach/sequence_reach_env.py', 'envs/factory.py',
               'simulation/motion/kinematics.py', 'simulation/physics/joint_control.py', 'simulation/physics/model.py')]
    return file_hashes(files, root=PROJECT_ROOT)


def parameter_hashes(model):
    groups={key:hashlib.sha256() for key in ('actor','critic','all')}
    for name,tensor in model.policy.state_dict().items():
        data=name.encode()+tensor.detach().cpu().contiguous().numpy().tobytes()
        groups['all'].update(data)
        if name.startswith(('mlp_extractor.policy_net.','action_net.','log_std')):groups['actor'].update(data)
        if name.startswith(('mlp_extractor.value_net.','value_net.')):groups['critic'].update(data)
    return {key:value.hexdigest() for key,value in groups.items()}


def optimizer_steps(model):
    # Adam's own per-parameter counters: different from PPO._n_updates (epochs).
    return max((int(state['step'].item()) for state in model.policy.optimizer.state.values() if 'step' in state),default=0)


def optimizer_state_hash(model):
    """Fingerprint Adam moments and parameter groups, including per-parameter steps."""
    import torch
    digest = hashlib.sha256()
    def update(value):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(repr(key).encode())
                update(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                update(item)
        elif torch.is_tensor(value):
            digest.update(str(value.dtype).encode()+str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    update(model.policy.optimizer.state_dict())
    return digest.hexdigest()

"""Reach provenance and shared model-state helpers; filesystem I/O lives in utils."""
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
              ('envs/reach_env.py', 'envs/sequence_reach_env.py', 'envs/factory.py',
               'simulation/kinematics.py', 'simulation/control.py', 'simulation/model.py')]
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

"""Strict one-time BC -> stock SAC deterministic mean-path initialization."""
import hashlib
from pathlib import Path

from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.torch_layers import CombinedExtractor

from .bc_reach import FEATURE_KEYS, load_policy, weight_digest
from .sac_reach import ReachTransform


def validate_compatibility(model, reference, transform):
    if reference.transform.config != transform.config:
        raise ValueError('BC preprocessing does not match SAC/environment preprocessing')
    observation = model.observation_space
    if not isinstance(observation, spaces.Dict) or tuple(observation.spaces) != FEATURE_KEYS:
        raise ValueError('BC/SAC Dict feature order differs')
    for key, shape in [('achieved_goal',(3,)), ('desired_goal',(3,)), ('observation',(27,))]:
        box = observation[key]
        if (not isinstance(box, spaces.Box) or box.shape != shape
                or box.dtype != np.dtype(transform.config['output_dtype'][key])
                or not np.all(np.isneginf(box.low)) or not np.all(np.isposinf(box.high))):
            raise ValueError('Incompatible processed observation space: '+key)
    action = model.action_space
    if (not isinstance(action, spaces.Box) or action.shape != (3,) or action.dtype != np.float32
            or not np.all(action.low == -1) or not np.all(action.high == 1)):
        raise ValueError('BC requires the original float32 Box(-1,1,(3,)) action space')
    actor = model.actor
    if (actor.use_sde or not actor.squash_output or actor.net_arch != [256,256]
            or actor.activation_fn is not nn.ReLU
            or type(actor.features_extractor) is not CombinedExtractor
            or tuple(actor.features_extractor.extractors) != FEATURE_KEYS
            or not all(type(e) is nn.Flatten for e in actor.features_extractor.extractors.values())
            or actor.features_extractor.features_dim != 33
            or type(actor.action_dist) is not SquashedDiagGaussianDistribution):
        raise ValueError('SAC feature extraction/activation/squashed Gaussian mean path differs from BC')
    expected = (nn.Linear, nn.ReLU, nn.Linear, nn.ReLU)
    for network in (actor, reference.network):
        if (tuple(type(m) for m in network.latent_pi) != expected
                or [(m.in_features,m.out_features) for m in network.latent_pi if type(m) is nn.Linear]
                != [(33,256),(256,256)] or type(network.mu) is not nn.Linear
                or (network.mu.in_features,network.mu.out_features) != (256,3)):
            raise ValueError('BC/SAC latent and output layer semantics differ')
    if type(actor.log_std) is not nn.Linear or (actor.log_std.in_features,actor.log_std.out_features) != (256,3):
        raise ValueError('Expected the unmodified stock non-SDE SAC log_std head')


def load_reference(model, path, transform):
    path = Path(path).resolve()
    reference, config = load_policy(path)  # weights_only; loader preserves Torch RNG
    validate_compatibility(model, reference, transform)
    for parameter in reference.network.parameters():
        parameter.requires_grad_(False)
    metadata = dict(path=str(path), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    network_sha256=weight_digest(reference.network),
                    observation_transform=config['observation_transform'],
                    bc_train_seeds=config['train_seeds'], bc_validation_seeds=config['validation_seeds'])
    return reference, metadata


def action_deviation(model, transform, reference, raw):
    """Deterministic actions only; never stochastic SAC samples or corrections."""
    if reference is None:
        return None
    expected = reference.predict(raw)
    actual, _ = model.predict(transform.encode(raw), deterministic=True)
    error = np.asarray(actual,dtype=float)-expected
    with torch.no_grad():
        tensors = {k:torch.as_tensor(v,device=model.device).float() for k,v in transform.encode(raw).items()}
        if tensors['observation'].ndim == 1:
            tensors = {k:v.unsqueeze(0) for k,v in tensors.items()}
        _, log_std, _ = model.actor.get_action_dist_params(tensors)
    return dict(samples=int(error.size//3), mse=float(np.mean(error**2)), mae=float(np.abs(error).mean()),
                max_abs=float(np.abs(error).max()), mean_policy_std=float(log_std.exp().mean().cpu()))


def initialize_actor_from_bc(model, path, transform, probe):
    """Only fresh SAC may be initialized. Optimizers retain their own parameters."""
    if (model.num_timesteps != 0 or model._n_updates != 0 or model.replay_buffer.size() != 0
            or model.actor.optimizer.state or model.critic.optimizer.state):
        raise ValueError('BC Actor initialization requires a fresh untrained SAC with empty replay/optimizer state')
    reference, metadata = load_reference(model, path, transform)
    protected = {name:weight_digest(module) for name,module in
                 [('log_std',model.actor.log_std),('critic',model.critic),('target_critic',model.critic_target)]}
    actor_before = weight_digest(model.actor)
    entropy_before = model.log_ent_coef.detach().clone() if model.log_ent_coef is not None else None
    for name in ('latent_pi','mu'):
        getattr(model.actor,name).load_state_dict(getattr(reference.network,name).state_dict(),strict=True)
    for name,module in [('log_std',model.actor.log_std),('critic',model.critic),('target_critic',model.critic_target)]:
        if weight_digest(module) != protected[name]:
            raise ValueError('BC initialization changed protected '+name)
    if entropy_before is not None and not torch.equal(entropy_before,model.log_ent_coef):
        raise ValueError('BC initialization changed entropy coefficient')
    deviation = action_deviation(model,transform,reference,probe)
    if deviation['max_abs'] > 1e-7:
        raise ValueError('BC/SAC deterministic actions differ beyond strict float32 tolerance')
    metadata.update(mapping={'BC.latent_pi.*':'SAC.actor.latent_pi.*','BC.mu.*':'SAC.actor.mu.*'},
                    actor_before_sha256=actor_before,actor_after_sha256=weight_digest(model.actor),
                    protected_sha256=protected,consistency=deviation,action_tolerance=1e-7,
                    log_std_initialization='Stock SB3 non-SDE nn.Linear weights/bias from the run seed; unchanged. '
                                           'BC features change its outputs, not its parameters; no constant std override.',
                    critic_initialization='Stock fresh double Critic; Target Critic initialized by SB3 copy, unchanged',
                    replay_prefill=False,optimizer_state_empty=True)
    return reference, metadata


def make_probe(environment, seeds):
    """Independent reset observations, not training rollouts or expert labels."""
    from ..envs.reach_env import ReachEnv
    with ReachEnv(**environment) as env:
        observations = [env.reset(seed=int(s))[0] for s in seeds]
        transform = ReachTransform.from_env(env)
    return {key:np.stack([obs[key] for obs in observations]) for key in FEATURE_KEYS},transform

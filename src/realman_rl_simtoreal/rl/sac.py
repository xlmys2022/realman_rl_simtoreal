"""Official SB3 SAC configuration, model construction and checkpoint format."""
import gzip
import io
from pathlib import Path
from stable_baselines3 import SAC

from .common.artifacts import write_json


SAC_CONFIG = dict(learning_rate=3e-4, batch_size=256, buffer_size=100_000,
                  learning_starts=100, gamma=.99, tau=.005, train_freq=1,
                  gradient_steps=1, ent_coef='auto', target_entropy='auto',
                  policy_kwargs=dict(net_arch=[256, 256]), device='cpu')


def make_model(env, seed, *, overrides=None):
    if env.observation_space['observation'].shape != (27,):
        raise ValueError("SAC experiments currently require observation_mode='legacy' (33 dimensions); use PPO for fetch_like_v1")
    params = dict(SAC_CONFIG)
    if overrides:
        params.update(overrides)
    return SAC('MultiInputPolicy', env, seed=seed, verbose=0, **params)


def save_checkpoint(model, folder, *, replay=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder/'model.zip')
    if replay:
        # SB3 accepts BytesIO, but does not accept gzip.GzipFile directly.
        serialized = io.BytesIO()
        model.save_replay_buffer(serialized)
        with gzip.open(folder/'replay_buffer.pkl.gz', 'wb') as file:
            file.write(serialized.getbuffer())
    write_json(folder/'state.json', dict(num_timesteps=int(model.num_timesteps), n_updates=int(model._n_updates),
                                      replay_size=int(model.replay_buffer.size())))

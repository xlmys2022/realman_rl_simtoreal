"""Algorithm-independent source and resource provenance for grasp experiments."""
import hashlib
from importlib.metadata import version
from ..paths import PROJECT_ROOT, GRASP_CONTACT_SCENE


def experiment_metadata():
    """Fingerprint the actual combined robot, gripper and contact task."""
    assets={GRASP_CONTACT_SCENE}
    for folder in ('assets/robots/rml63','assets/grippers/eg2_4c2'):
        assets.update(p for p in (PROJECT_ROOT/folder).rglob('*') if p.is_file())
    sources=[PROJECT_ROOT/'src/realman_rl_simtoreal'/p for p in (
        'envs/grasp_lift_env.py','tasks/grasp_lift.py','evaluation/grasp_lift.py','utils/artifacts.py','evaluation/metrics.py',
        'simulation/grasp_controller.py','simulation/gripper.py','simulation/grasp_contact.py',
        'simulation/control.py','simulation/kinematics.py','simulation/model.py')]
    def hashes(paths):
        return {str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return dict(assets=hashes(assets),sources=hashes(sources),
                runtime={name:version(name) for name in ('mujoco','numpy','gymnasium')},
                action_semantics=['world_dx','world_dy','world_dz','absolute_gripper_opening'],
                observation_dim=41,action_dim=4)

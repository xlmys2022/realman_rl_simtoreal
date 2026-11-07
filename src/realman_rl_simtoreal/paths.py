"""Paths for the supported source-checkout editable installation.

Assets are deliberately kept in the repository, not bundled in a wheel. Keep
the checkout available after installation. No path depends on the working dir.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ASSETS = PROJECT_ROOT / "assets/robots/rml63"
DEFAULT_SCENE = ROBOT_ASSETS / "scene.xml"
DEFAULT_URDF = ROBOT_ASSETS / "rml63_mujoco.urdf"
OUTPUTS = PROJECT_ROOT / "outputs"
GRIPPER_ASSETS = PROJECT_ROOT / "assets/grippers/eg2_4c2"
GRIPPER_SCENE = GRIPPER_ASSETS / "rml63_eg2_4c2.xml"
GRASP_CONTACT_SCENE = PROJECT_ROOT / "assets/scenes/grasp_contact.xml"

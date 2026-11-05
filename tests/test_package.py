"""Editable-package, isolated-process and existing-entry compatibility checks."""

import ast
from contextlib import contextmanager
from importlib import metadata, util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from realman_rl_simtoreal.paths import PROJECT_ROOT as ROOT


class PackageTests(unittest.TestCase):
    def run_code(self, code):
        with tempfile.TemporaryDirectory(prefix="realman-import-") as directory:
            env = dict(os.environ, PYTHONPATH="/nonexistent/ros/python")
            result = subprocess.run([sys.executable, "-I", "-c", code], cwd=directory,
                                    env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result

    def test_package_initializers_are_lightweight(self):
        self.run_code('''
import sys
import realman_rl_simtoreal as package
import realman_rl_simtoreal.simulation
import realman_rl_simtoreal.vendor
import realman_rl_simtoreal.experiments
assert '/src/realman_rl_simtoreal/' in package.__file__
assert not any(n in sys.modules for n in ('mujoco','glfw','Robotic_Arm'))
''')

    def test_core_import_and_resources_under_isolated_python(self):
        self.run_code('''
import os,sys
os.environ.pop('PYGLFW_LIBRARY_VARIANT',None)
before=os.environ.get('PYGLFW_LIBRARY_VARIANT')
from realman_rl_simtoreal.paths import PROJECT_ROOT, DEFAULT_SCENE, DEFAULT_URDF
from realman_rl_simtoreal.simulation.model import load_model, initialize
from realman_rl_simtoreal.simulation.kinematics import Kinematics
assert sys.flags.isolated and '/nonexistent/ros/python' not in sys.path
assert DEFAULT_SCENE.is_file() and DEFAULT_URDF.is_file()
assert (PROJECT_ROOT/'pyproject.toml').is_file()
for enabled in (False,True):
    model,jids,aids=load_model(gravity_compensation=enabled)
    data=initialize(model); kin=Kinematics(model)
    assert (model.nq,model.nv,model.nu)==(6,6,6) and data.time==0
    assert kin.solve_ik(kin.home_position+[.005,0,0],kin.home_q).success
assert os.environ.get('PYGLFW_LIBRARY_VARIANT')==before
assert 'mujoco.viewer' not in sys.modules and 'Robotic_Arm' not in sys.modules
assert not any(n in sys.modules for n in ('control_joints','control_ee','kinematics'))
''')

    def test_simulation_and_adapter_imports_with_sdk_blocked(self):
        self.run_code('''
import importlib.abc,sys
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='Robotic_Arm' or fullname.startswith('Robotic_Arm.'):
            raise ModuleNotFoundError('SDK deliberately unavailable')
sys.meta_path.insert(0,BlockSDK())
from realman_rl_simtoreal.simulation.model import load_model,initialize
from realman_rl_simtoreal.simulation.control import run_segment
from realman_rl_simtoreal.simulation.kinematics import Kinematics
from realman_rl_simtoreal.simulation.ee_control import check_motion
from realman_rl_simtoreal.vendor.realman_fk import RealManFK,SDKUnavailable
from realman_rl_simtoreal.vendor.realman_ik import RealManIK
model,jids,aids=load_model(gravity_compensation=True)
data=initialize(model); kin=Kinematics(model)
target=kin.home_position+[.005,0,0]
result=kin.solve_ik(target,kin.home_q)
assert result.success
check_motion(model,data,jids,aids,kin,result,target)
run_segment(model,data,jids,aids,kin.home_q,'SDK absent',ramp=0,hold=.02)
assert data.time>0
for constructor in (lambda:RealManFK(),lambda:RealManIK(kin.limits)):
    try: constructor()
    except SDKUnavailable: pass
    else: raise AssertionError('SDK constructor must report unavailability')
''')

    def test_metadata_uses_existing_dependency_files_and_optional_sdk(self):
        def requirements(name):
            return {line.strip() for line in (ROOT/name).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')}
        installed = metadata.metadata('realman_rl_simtoreal')
        self.assertEqual(installed['Version'], '0.1.0')
        self.assertEqual(installed['Requires-Python'], '>=3.11')
        requires = installed.get_all('Requires-Dist')
        self.assertEqual({r for r in requires if ';' not in r}, requirements('requirements.txt'))
        optional = [r for r in requires if ';' in r]
        self.assertEqual({r.split(';')[0].strip() for r in optional}, requirements('requirements-vendor.txt'))
        self.assertTrue(all('extra == "vendor"' in r for r in optional))
        direct = json.loads(metadata.distribution('realman_rl_simtoreal').read_text('direct_url.json'))
        self.assertTrue(direct['dir_info']['editable'])

    def test_canonical_import_graph_has_no_scripts_or_cycles(self):
        package = ROOT/'src/realman_rl_simtoreal'
        modules = {}
        for path in package.rglob('*.py'):
            parts = list(path.relative_to(package.parent).with_suffix('').parts)
            if parts[-1]=='__init__': parts.pop()
            modules['.'.join(parts)] = path
        edges = {name:set() for name in modules}
        for name,path in modules.items():
            current_package = name if path.name=='__init__.py' else name.rpartition('.')[0]
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.Import):
                    imports = [a.name for a in node.names]
                elif isinstance(node,ast.ImportFrom):
                    base = util.resolve_name('.'*node.level+(node.module or ''),current_package) if node.level else node.module
                    imports = [base]+[base+'.'+a.name for a in node.names]
                else:
                    continue
                self.assertFalse(any(n=='scripts' or n.startswith('scripts.') for n in imports))
                if '.simulation.' in name:
                    self.assertFalse(any(n.startswith(('realman_rl_simtoreal.vendor','realman_rl_simtoreal.experiments',
                                                       'realman_rl_simtoreal.viewer')) for n in imports))
                edges[name].update(n for n in imports if n in modules)
        done,visiting=set(),set()
        def visit(name):
            self.assertNotIn(name,visiting,'Circular package import')
            if name in done:return
            visiting.add(name)
            for dep in edges[name]:visit(dep)
            visiting.remove(name);done.add(name)
        for name in edges:visit(name)
        for path in [*package.rglob('*.py'),*(ROOT/'scripts').glob('*.py'),*(ROOT/'tests').glob('*.py')]:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertFalse(ast.unparse(node.func).startswith('sys.path.'),str(path))
        for old in ('kinematics.py','realman_fk.py','realman_ik.py'):
            self.assertFalse((ROOT/'scripts'/old).exists())

    def test_existing_argparse_entries_help_from_other_directory(self):
        names = ('inspect_model','view_robot','control_joints','control_ee','compare_controllers','compare_fk','compare_ik')
        for name in names:
            with self.subTest(script=name):
                result = subprocess.run([sys.executable,'-I',str(ROOT/'scripts'/f'{name}.py'),'--help'],
                                        cwd='/tmp',capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('usage:',result.stdout)

    def test_cli_failure_and_argument_exit_semantics(self):
        cases = [('inspect_model',['--model','/tmp/realman-missing.urdf'],1),
                 ('control_joints',['--duration','1'],2),('control_ee',['--duration','1'],2),
                 ('compare_fk',['--model','/tmp/realman-missing.xml'],1),
                 ('compare_ik',['--model','/tmp/realman-missing.xml'],1)]
        for name,args,code in cases:
            with self.subTest(script=name):
                result=subprocess.run([sys.executable,'-I',str(ROOT/'scripts'/f'{name}.py'),*args],
                                      cwd='/tmp',capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,code,result.stdout+result.stderr)

    def test_static_viewer_missing_display_fails_clearly(self):
        env={k:v for k,v in os.environ.items() if k not in ('DISPLAY','WAYLAND_DISPLAY')}
        result=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/view_robot.py'),'--duration','1'],
                              cwd='/tmp',env=env,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,1)
        self.assertIn('No desktop display',result.stderr)

    def test_viewer_backend_defaults_and_explicit_preference(self):
        from realman_rl_simtoreal.viewer import configure_backend
        with patch.dict(os.environ,{'DISPLAY':':0'},clear=True):
            configure_backend()
            self.assertEqual(os.environ['PYGLFW_LIBRARY_VARIANT'],'x11')
        with patch.dict(os.environ,{'DISPLAY':':0','PYGLFW_LIBRARY_VARIANT':'wayland'},clear=True):
            configure_backend()
            self.assertEqual(os.environ['PYGLFW_LIBRARY_VARIANT'],'wayland')

    def test_viewer_threads_are_joined_on_normal_and_exceptional_exit(self):
        import mujoco
        from realman_rl_simtoreal.viewer import passive_viewer
        handle=object()
        @contextmanager
        def launch(*args):
            yield handle
        fake=SimpleNamespace(launch_passive=launch)
        for exceptional in (False,True):
            existing,worker=Mock(),Mock()
            with (patch.dict(sys.modules,{'mujoco.viewer':fake}),
                  patch.object(mujoco,'viewer',fake,create=True),
                  patch('realman_rl_simtoreal.viewer.threading.enumerate',side_effect=[[existing],[existing,worker]])):
                try:
                    with passive_viewer(None,None) as viewer:
                        self.assertIs(viewer,handle)
                        if exceptional:raise InterruptedError('Viewer closed')
                except InterruptedError:
                    self.assertTrue(exceptional)
            worker.join.assert_called_once_with()
            existing.join.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)

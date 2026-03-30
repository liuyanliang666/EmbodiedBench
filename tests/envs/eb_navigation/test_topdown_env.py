import importlib
import sys
import types
import unittest
from unittest import mock


def _install_stub_dependencies():
    gym_mod = types.ModuleType("gym")

    class DummyEnv:
        pass

    class DummyDiscrete:
        def __init__(self, n):
            self.n = n

    gym_mod.Env = DummyEnv
    gym_mod.spaces = types.SimpleNamespace(Discrete=DummyDiscrete)
    sys.modules["gym"] = gym_mod

    numpy_mod = types.ModuleType("numpy")
    numpy_mod.array = lambda value: value
    numpy_mod.asarray = lambda value, dtype=None: value
    numpy_mod.float32 = float
    numpy_mod.int32 = int
    sys.modules["numpy"] = numpy_mod

    class DummyEvent:
        def __init__(self):
            self.frame = [[1]]
            self.depth_frame = [[1000]]
            self.metadata = {
                "agent": {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": {"y": 0.0},
                    "cameraHorizon": 0.0,
                },
                "lastActionSuccess": True,
                "errorMessage": "",
                "lastAction": "Teleport",
            }
            self.third_party_camera_frames = []

    class DummyController:
        def __init__(self, **_kwargs):
            self.last_event = DummyEvent()

        def reset(self, **_kwargs):
            self.last_event = DummyEvent()
            return self.last_event

        def step(self, *args, **kwargs):
            self.last_event = DummyEvent()
            action = kwargs.get("action")
            if action == "GetMapViewCameraProperties":
                self.last_event.metadata["actionReturn"] = {}
            elif action == "GetReachablePositions":
                self.last_event.metadata["actionReturn"] = [{"x": 0.0, "y": 0.9, "z": 0.0}]
            return self.last_event

        def stop(self):
            return None

    ai2thor_mod = types.ModuleType("ai2thor")
    controller_mod = types.ModuleType("ai2thor.controller")
    controller_mod.Controller = DummyController
    platform_mod = types.ModuleType("ai2thor.platform")
    platform_mod.CloudRendering = object()
    ai2thor_mod.controller = controller_mod
    ai2thor_mod.platform = platform_mod
    sys.modules["ai2thor"] = ai2thor_mod
    sys.modules["ai2thor.controller"] = controller_mod
    sys.modules["ai2thor.platform"] = platform_mod

    utils_mod = types.ModuleType("embodiedbench.envs.eb_navigation.utils")
    utils_mod.draw_target_box = lambda *args, **kwargs: None
    utils_mod.draw_boxes = lambda *args, **kwargs: None
    sys.modules["embodiedbench.envs.eb_navigation.utils"] = utils_mod

    pil_mod = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.fromarray = lambda value: types.SimpleNamespace(save=lambda *_args, **_kwargs: None)
    pil_mod.Image = image_mod
    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = image_mod

    main_mod = types.ModuleType("embodiedbench.main")
    main_mod.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    sys.modules["embodiedbench.main"] = main_mod


_install_stub_dependencies()
sys.modules.pop("embodiedbench.envs.eb_navigation.EBNavEnv", None)
nav_env_module = importlib.import_module("embodiedbench.envs.eb_navigation.EBNavEnv")
EBNavigationEnv = nav_env_module.EBNavigationEnv


class DummyTopdownBuilder:
    def __init__(self):
        self.calls = 0

    def add_event(self, _event):
        self.calls += 1
        return [[self.calls]]


class NavigationTopdownEnvTests(unittest.TestCase):
    @mock.patch.object(EBNavigationEnv, "_load_dataset", return_value=[
        {
            "instruction": "Find the mug.",
            "scene": "FloorPlan1",
            "agentPose": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": 0.0,
                "horizon": 0.0,
            },
            "targetObjectIds": "Mug|1",
            "target_position": {"x": 0.0, "z": 0.0},
        }
    ])
    def test_reset_returns_topdown_rgb_when_enabled(self, _load_dataset):
        with mock.patch.object(EBNavigationEnv, "_build_topdown_builder", return_value=DummyTopdownBuilder()):
            env = EBNavigationEnv(
                eval_set="base",
                exp_name="test",
                use_topdown_prompt=True,
            )

            obs = env.reset()

        self.assertIn("head_rgb", obs)
        self.assertIn("topdown_rgb", obs)
        self.assertEqual(obs["topdown_rgb"], [[1]])


if __name__ == "__main__":
    unittest.main()

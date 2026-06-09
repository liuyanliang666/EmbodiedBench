import importlib
import sys
import types
import unittest


def _install_evaluator_stub_dependencies(captured):
    env_mod = types.ModuleType("embodiedbench.envs.eb_alfred.EBAlfEnv")

    class DummyEnv:
        def __init__(self, *args, **kwargs):
            self.language_skill_set = ["MoveAhead"]
            self.log_path = "/private/tmp"
            self.number_of_episodes = 0
            self._current_episode_num = 0

        def close(self):
            pass

    env_mod.EBAlfEnv = DummyEnv
    env_mod.ValidEvalSets = ["base"]
    sys.modules["embodiedbench.envs.eb_alfred.EBAlfEnv"] = env_mod

    planner_mod = types.ModuleType("embodiedbench.planner.vlm_planner")

    class DummyPlanner:
        def __init__(self, *args, **kwargs):
            captured["planner_kwargs"] = kwargs

    planner_mod.VLMPlanner = DummyPlanner
    sys.modules["embodiedbench.planner.vlm_planner"] = planner_mod

    summarize_mod = types.ModuleType("embodiedbench.evaluator.summarize_result")
    summarize_mod.average_json_values = lambda *args, **kwargs: None
    sys.modules["embodiedbench.evaluator.summarize_result"] = summarize_mod

    main_mod = types.ModuleType("embodiedbench.main")
    main_mod.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    sys.modules["embodiedbench.main"] = main_mod


class EBAlfredEvaluatorConfigTests(unittest.TestCase):
    def test_passes_prompt_template_path_to_planner_kwargs(self):
        captured = {}
        _install_evaluator_stub_dependencies(captured)
        sys.modules.pop("embodiedbench.evaluator.eb_alfred_evaluator", None)
        evaluator_module = importlib.import_module("embodiedbench.evaluator.eb_alfred_evaluator")

        config = {
            "model_name": "Qwen2.5-VL-7B-Instruct",
            "model_type": "vllm_direct",
            "down_sample_ratio": 1,
            "language_only": False,
            "eval_sets": ["base"],
            "chat_history": False,
            "n_shots": 0,
            "detection_box": False,
            "multistep": False,
            "resolution": 600,
            "exp_name": "test",
            "env_feedback": True,
            "tp": 1,
            "easyr1_format": True,
            "max_episode_steps": 30,
            "max_invalid_actions": 10,
            "prompt_template_path": "/tmp/custom_alfred.jinja",
            "point_input_resolution": 1000,
            "point_output_resolution": 300,
        }

        evaluator = evaluator_module.EB_AlfredEvaluator(config)
        evaluator.evaluate_main()

        self.assertEqual(
            captured["planner_kwargs"]["kwargs"]["prompt_template_path"],
            "/tmp/custom_alfred.jinja",
        )
        self.assertEqual(captured["planner_kwargs"]["kwargs"]["point_input_resolution"], 1000)
        self.assertEqual(captured["planner_kwargs"]["kwargs"]["point_output_resolution"], 300)


if __name__ == "__main__":
    unittest.main()

import importlib
import os
import sys
import types
import unittest


def _install_planner_stub_dependencies():
    torch_mod = types.ModuleType("torch")
    sys.modules["torch"] = torch_mod

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.imwrite = lambda *_args, **_kwargs: True
    sys.modules["cv2"] = cv2_mod

    numpy_mod = types.ModuleType("numpy")
    numpy_mod.array_equal = lambda left, right: left == right
    numpy_mod.random = types.SimpleNamespace(randint=lambda *_args, **_kwargs: 0)
    sys.modules["numpy"] = numpy_mod

    planner_utils_mod = types.ModuleType("embodiedbench.planner.planner_utils")
    planner_utils_mod.local_image_to_data_url = lambda image_path: f"url::{os.path.basename(image_path)}"
    planner_utils_mod.template = ""
    planner_utils_mod.template_lang = ""
    planner_utils_mod.fix_json = lambda value: value
    sys.modules["embodiedbench.planner.planner_utils"] = planner_utils_mod

    remote_model_mod = types.ModuleType("embodiedbench.planner.remote_model")

    class DummyRemoteModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def respond(self, *_args, **_kwargs):
            return '[{"action_type": "Move", "parameter": "forward"}]'

    remote_model_mod.RemoteModel = DummyRemoteModel
    sys.modules["embodiedbench.planner.remote_model"] = remote_model_mod

    custom_model_mod = types.ModuleType("embodiedbench.planner.custom_model")

    class DummyCustomModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def respond(self, *_args, **_kwargs):
            return '[{"action_type": "Move", "parameter": "forward"}]'

    custom_model_mod.CustomModel = DummyCustomModel
    sys.modules["embodiedbench.planner.custom_model"] = custom_model_mod

    generation_guide_mod = types.ModuleType("embodiedbench.planner.planner_config.generation_guide")
    generation_guide_mod.llm_generation_guide = {}
    generation_guide_mod.vlm_generation_guide = {}
    sys.modules["embodiedbench.planner.planner_config.generation_guide"] = generation_guide_mod

    main_mod = types.ModuleType("embodiedbench.main")
    main_mod.logger = types.SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    sys.modules["embodiedbench.main"] = main_mod


_install_planner_stub_dependencies()
sys.modules.pop("embodiedbench.planner.vlm_planner", None)
vlm_planner_module = importlib.import_module("embodiedbench.planner.vlm_planner")
VLMPlanner = vlm_planner_module.VLMPlanner


class AlfredTopdownPlannerTests(unittest.TestCase):
    def test_navigation_prompt_includes_image_placeholder_after_task(self):
        planner = VLMPlanner(
            "dummy-model",
            "custom",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            kwargs={"easyr1_variant": "navigation"},
        )

        prompt = planner.process_prompt("Find a mug.")

        self.assertIn("Task: Find a mug\n<image>", prompt)
        self.assertIn(
            "You are a household navigation robot. Given the current first-person photo and the task below, your goal is to predict the next optimal discrete navigation action.",
            prompt,
        )

    def test_single_image_prompt_matches_easyr1_style(self):
        planner = VLMPlanner(
            "dummy-model",
            "custom",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            use_topdown_prompt=False,
        )

        prompt = planner.process_prompt("Find a mug.")

        self.assertIn(
            "Given the current first-person photo and the task below, your goal is to predict the next optimal discrete action.",
            prompt,
        )
        self.assertIn("Task: Find a mug.\n<image>", prompt)
        self.assertNotIn("Current first-person view:", prompt)
        self.assertNotIn("The second image is a reconstructed top-down occupancy map", prompt)

    def test_topdown_prompt_uses_local_template(self):
        planner = VLMPlanner(
            "dummy-model",
            "custom",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=False,
            use_topdown_prompt=True,
        )

        prompt = planner.process_prompt("Find a mug.")

        self.assertTrue(planner.use_easyr1_format)
        self.assertIn("The first image is the current first-person photo.", prompt)
        self.assertIn("The second image is a reconstructed top-down occupancy map", prompt)
        self.assertIn("Task: Find a mug", prompt)
        self.assertIn("Current first-person view:\n<image>", prompt)
        self.assertIn("Reconstructed top-down occupancy map from the trajectory observed so far:\n<image>", prompt)

    def test_navigation_topdown_prompt_uses_navigation_only_template(self):
        planner = VLMPlanner(
            "dummy-model",
            "custom",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            use_topdown_prompt=True,
            kwargs={"easyr1_variant": "navigation"},
        )

        prompt = planner.process_prompt("Find a mug.")

        self.assertIn(
            "Given the current first-person photo, the reconstructed top-down occupancy map, and the task below, your goal is to predict the next optimal discrete navigation action.",
            prompt,
        )
        self.assertIn("Task: Find a mug.\nCurrent first-person view:\n<image>", prompt)
        self.assertIn("Reconstructed top-down occupancy map from the trajectory observed so far:\n<image>", prompt)
        self.assertNotIn("Interaction Actions (Manipulating objects):", prompt)
        self.assertNotIn("PickupObject", prompt)

    def test_get_message_with_topdown_adds_two_images_in_order(self):
        planner = VLMPlanner(
            "dummy-model",
            "custom",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            use_topdown_prompt=True,
        )

        messages = planner.get_message(
            {
                "head_rgb": "/tmp/head.png",
                "topdown_rgb": "/tmp/topdown.png",
            },
            "prompt text",
        )

        content = messages[0]["content"]
        image_urls = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
        self.assertEqual(image_urls, ["url::head.png", "url::topdown.png"])
        self.assertEqual(content[-1]["type"], "text")
        self.assertEqual(content[-1]["text"], "prompt text")

    def test_custom_model_receives_two_images_when_topdown_enabled(self):
        planner = VLMPlanner(
            "dummy-model",
            "custom",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            use_topdown_prompt=True,
        )
        captured = {}

        def fake_respond(prompt, obs=None):
            captured["prompt"] = prompt
            captured["obs"] = obs
            return '[{"action_type": "Move", "parameter": "forward"}]'

        planner.model.respond = fake_respond

        action, _ = planner.act(
            {
                "head_rgb": "/tmp/head.png",
                "topdown_rgb": "/tmp/topdown.png",
            },
            "Find a mug",
        )

        self.assertEqual(action, "MoveAhead")
        self.assertEqual(captured["obs"], ["/tmp/head.png", "/tmp/topdown.png"])


if __name__ == "__main__":
    unittest.main()

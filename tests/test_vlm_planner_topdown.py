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
            "Given the current first-person photo, your history summary, and the task below, your goal is to predict the next optimal discrete action and report the current history summary.",
            prompt,
        )
        self.assertIn("Task: Find a mug\n<image>", prompt)
        self.assertNotIn("Current first-person view:", prompt)
        self.assertNotIn("the reconstructed top-down occupancy map", prompt)
        self.assertNotIn("top-down arrow", prompt)
        self.assertIn("Held object: nothing", prompt)
        self.assertIn("Last action: none (first step)", prompt)
        self.assertIn("History summary:\n(none, first step)", prompt)
        self.assertIn("<summary>", prompt)
        self.assertIn("<progress>", prompt)
        self.assertIn("<spatial>", prompt)

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
        self.assertIn(
            "Given the current first-person photo",
            prompt,
        )
        self.assertIn("the reconstructed top-down occupancy map", prompt)
        self.assertIn("your history summary, and the task below", prompt)
        self.assertIn("The first image is the current first-person photo.", prompt)
        self.assertIn("The second image is a reconstructed top-down occupancy map", prompt)
        self.assertIn("Task: Find a mug", prompt)
        self.assertIn("Current first-person view:\n<image>", prompt)
        self.assertIn("Reconstructed top-down occupancy map from the trajectory observed so far:\n<image>", prompt)
        self.assertIn("top-down arrow", prompt)
        self.assertIn("Held object: nothing", prompt)
        self.assertIn("Last action: none (first step)", prompt)
        self.assertIn("History summary:\n(none, first step)", prompt)

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

    def test_get_message_vllm_direct_keeps_raw_single_image_object(self):
        planner = VLMPlanner(
            "dummy-model",
            "vllm_direct",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            use_topdown_prompt=False,
        )

        raw_image = [[1, 2], [3, 4]]
        messages = planner.get_message(raw_image, "Task: test.\n<image>")

        content = messages[0]["content"]
        image_items = [item for item in content if item["type"] == "image"]
        self.assertEqual(len(image_items), 1)
        self.assertIs(image_items[0]["image"], raw_image)
        self.assertFalse(any(item["type"] == "image_url" for item in content))

    def test_get_message_vllm_direct_trims_trailing_newline_on_last_text_segment(self):
        planner = VLMPlanner(
            "dummy-model",
            "vllm_direct",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            use_topdown_prompt=False,
        )

        raw_image = [[1, 2], [3, 4]]
        messages = planner.get_message(raw_image, "Task: test.\n<image>\nOutput one action.\n")

        content = messages[0]["content"]
        text_items = [item["text"] for item in content if item["type"] == "text"]
        self.assertEqual(text_items[-1], "\nOutput one action.")

    def test_get_message_vllm_direct_topdown_keeps_raw_image_objects(self):
        planner = VLMPlanner(
            "dummy-model",
            "vllm_direct",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            use_topdown_prompt=True,
        )

        head = [[1]]
        topdown = [[2]]
        messages = planner.get_message(
            {
                "head_rgb": head,
                "topdown_rgb": topdown,
            },
            "Task: test.\nCurrent first-person view:\n<image>\nReconstructed top-down occupancy map from the trajectory observed so far:\n<image>",
        )

        content = messages[0]["content"]
        image_items = [item for item in content if item["type"] == "image"]
        self.assertEqual([item["image"] for item in image_items], [head, topdown])
        self.assertFalse(any(item["type"] == "image_url" for item in content))

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


class AlfredSummaryRecursionTests(unittest.TestCase):
    """Cover the EasyR1 v4-summary fields injected into alfred.jinja."""

    def _make_planner(self, use_topdown_prompt=False, kwargs=None):
        return VLMPlanner(
            "dummy-model",
            "custom",
            actions=["MoveAhead"],
            system_prompt="",
            examples=[],
            use_easyr1_format=True,
            use_topdown_prompt=use_topdown_prompt,
            kwargs=kwargs or {},
        )

    def test_held_object_extracted_from_recent_feedback(self):
        planner = self._make_planner()
        feedback = [[None, "Action executed. Currently holding: Apple. Other info.", 1.0]]
        prompt = planner.process_prompt("Find a mug.", prev_act_feedback=feedback)
        # Training data sources held_object from PDDL args (lowercase like
        # "apple"), so we lowercase the THOR PascalCase objectType on the way
        # into the prompt to keep eval-time inputs in distribution.
        self.assertIn("Held object: apple", prompt)
        self.assertNotIn("Held object: Apple", prompt)
        self.assertNotIn("Held object: nothing", prompt)

    def test_held_object_lowercases_compound_thor_name(self):
        planner = self._make_planner()
        feedback = [[None, "Currently holding: FloorLamp.", 1.0]]
        prompt = planner.process_prompt("Find a mug.", prev_act_feedback=feedback)
        self.assertIn("Held object: floorlamp", prompt)

    def test_held_object_explicit_nothing_propagates(self):
        planner = self._make_planner()
        feedback = [[None, "Currently holding: nothing.", 1.0]]
        prompt = planner.process_prompt("Find a mug.", prev_act_feedback=feedback)
        self.assertIn("Held object: nothing", prompt)

    def test_last_action_uses_v4_format_after_act(self):
        planner = self._make_planner()

        def fake_respond(prompt, obs=None):
            return (
                "<summary>\n"
                "<progress>\nProgressing.\n</progress>\n"
                "<spatial>\nLayout note.\n</spatial>\n"
                "</summary>\n"
                "<think> reasoning </think>\n"
                '<answer> [{"action_type": "PickupObject", "parameter": [305, 295]}] </answer>'
            )

        planner.model.respond = fake_respond
        planner.act("/tmp/head.png", "Pick the apple")

        prompt = planner.process_prompt("Pick the apple", prev_act_feedback=[])
        self.assertIn("Last action: PickupObject at [305, 295]", prompt)

    def test_prev_summary_is_recursed_into_next_prompt(self):
        planner = self._make_planner()

        def fake_respond(prompt, obs=None):
            return (
                "<summary>\n"
                "<progress>\nI have located the apple.\n</progress>\n"
                "<spatial>\nThe apple sits on the counter at front-right.\n</spatial>\n"
                "</summary>\n"
                "<think> ... </think>\n"
                '<answer> [{"action_type": "Move", "parameter": "forward"}] </answer>'
            )

        planner.model.respond = fake_respond
        planner.act("/tmp/head.png", "Pick the apple")

        prompt = planner.process_prompt("Pick the apple", prev_act_feedback=[])
        self.assertIn("History summary:\n<summary>", prompt)
        self.assertIn("I have located the apple.", prompt)
        self.assertIn("The apple sits on the counter at front-right.", prompt)
        self.assertNotIn("(none, first step)", prompt)

    def test_backward_outputs_are_invalid_in_easyr1_alfred_mode(self):
        planner = self._make_planner()

        backward_outputs = [
            '<answer> [{"action_type": "Move", "parameter": "backward"}] </answer>',
            '<answer> MoveBack </answer>',
            '<answer> Move backward by 0.25 </answer>',
        ]

        for output in backward_outputs:
            with self.subTest(output=output):
                self.assertEqual(planner.json_to_action(output), -1)

    def test_qwen3_prompt_template_scales_normalized_points_to_image_coordinates(self):
        planner = self._make_planner(
            kwargs={"prompt_template_path": "/tmp/alfred_summary_v4_concise_qwen3.jinja"}
        )

        action = planner.json_to_action(
            '<answer> [{"action_type": "PickupObject", "parameter": [508, 492]}] </answer>'
        )

        self.assertEqual(action, {"action": "pickup_by_point", "point": [305, 295]})

    def test_qwen3_point_output_resolution_can_target_interaction_frame(self):
        planner = self._make_planner(
            kwargs={
                "prompt_template_path": "/tmp/alfred_summary_v4_concise_qwen3.jinja",
                "point_output_resolution": 300,
            }
        )

        action = planner.json_to_action(
            '<answer> [{"action_type": "PickupObject", "parameter": [508, 492]}] </answer>'
        )

        self.assertEqual(action, {"action": "pickup_by_point", "point": [152, 148]})

    def test_first_step_defaults_when_no_prior_state(self):
        planner = self._make_planner()
        prompt = planner.process_prompt("Find a mug.", prev_act_feedback=[])
        self.assertIn("Held object: nothing", prompt)
        self.assertIn("Last action: none (first step)", prompt)
        self.assertIn("History summary:\n(none, first step)", prompt)
        self.assertNotIn("previous action was invalid and did not change the scene", prompt)

    def test_stuck_hint_appended_when_frames_unchanged(self):
        planner = self._make_planner()
        same_obs = [[1, 2], [3, 4]]
        planner.update_info(
            {"env_feedback": "Currently holding: nothing.", "action_id": 0},
            previous_obs=same_obs,
            current_obs=same_obs,
        )
        prompt = planner.process_prompt("Find a mug.", prev_act_feedback=planner.episode_act_feedback)
        self.assertIn("previous action was invalid and did not change the scene", prompt)
        # Hint must come AFTER the rendered template, not in the middle.
        self.assertTrue(
            prompt.rstrip().endswith("viewpoint, or prerequisite was likely incorrect."),
            "Stuck hint should be appended at the end of the prompt.",
        )

    def test_stuck_hint_clears_when_frames_change(self):
        planner = self._make_planner()
        # First update flips it on.
        same_obs = [[1, 2], [3, 4]]
        planner.update_info(
            {"env_feedback": "Currently holding: nothing.", "action_id": 0},
            previous_obs=same_obs,
            current_obs=same_obs,
        )
        self.assertTrue(planner.frames_unchanged_hint)
        # Second update with different obs should clear it.
        planner.update_info(
            {"env_feedback": "Currently holding: nothing.", "action_id": 0},
            previous_obs=[[1, 2], [3, 4]],
            current_obs=[[9, 9], [9, 9]],
        )
        prompt = planner.process_prompt("Find a mug.", prev_act_feedback=planner.episode_act_feedback)
        self.assertNotIn("previous action was invalid and did not change the scene", prompt)

    def test_format_action_description_matches_v4_helper(self):
        # Mirrors generate_summaries_v4.get_action_description so eval-time
        # last_action strings stay in distribution with training.
        self.assertEqual(
            VLMPlanner._format_action_description(
                {"action_type": "PickupObject", "parameter": [305, 295]}
            ),
            "PickupObject at [305, 295]",
        )
        self.assertEqual(
            VLMPlanner._format_action_description({"action_type": "Move", "parameter": "forward"}),
            "Move forward",
        )
        self.assertEqual(
            VLMPlanner._format_action_description({"action_type": "Look"}),
            "Look",
        )

    def test_extract_summary_block_preserves_tags(self):
        out = (
            "<summary>\n"
            "<progress>\nSeen the apple.\n</progress>\n"
            "<spatial>\nApple on the counter.\n</spatial>\n"
            "</summary>\n"
            "<think>...</think><answer>x</answer>"
        )
        block = VLMPlanner._extract_summary_block(out)
        self.assertTrue(block.startswith("<summary>"))
        self.assertTrue(block.endswith("</summary>"))
        self.assertIn("Seen the apple.", block)

    def test_topdown_prompt_renders_summary_recursion_fields(self):
        planner = self._make_planner(use_topdown_prompt=True)

        def fake_respond(prompt, obs=None):
            return (
                "<summary>\n"
                "<progress>\nI just rotated.\n</progress>\n"
                "<spatial>\nKitchen on my left.\n</spatial>\n"
                "</summary>\n"
                "<think>...</think>\n"
                '<answer> [{"action_type": "Rotate", "parameter": "left"}] </answer>'
            )

        planner.model.respond = fake_respond
        planner.act({"head_rgb": "/tmp/h.png", "topdown_rgb": "/tmp/t.png"}, "Find a mug")

        prompt = planner.process_prompt("Find a mug", prev_act_feedback=[])
        self.assertIn("Last action: Rotate left", prompt)
        self.assertIn("I just rotated.", prompt)
        self.assertIn("Kitchen on my left.", prompt)
        self.assertIn("the reconstructed top-down occupancy map", prompt)


if __name__ == "__main__":
    unittest.main()

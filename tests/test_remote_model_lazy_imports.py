import os
import subprocess
import sys
import textwrap
import unittest


class RemoteModelLazyImportTests(unittest.TestCase):
    def test_remote_model_import_does_not_require_lmdeploy(self):
        script = textwrap.dedent(
            """
            import importlib
            import sys
            import types

            class BlockLmdeployImporter:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "lmdeploy" or fullname.startswith("lmdeploy."):
                        raise ImportError("lmdeploy is intentionally unavailable")
                    return None

            anthropic_mod = types.ModuleType("anthropic")
            anthropic_mod.Anthropic = type("DummyAnthropic", (), {})
            sys.modules["anthropic"] = anthropic_mod

            google_mod = types.ModuleType("google")
            google_genai_mod = types.ModuleType("google.generativeai")
            google_mod.generativeai = google_genai_mod
            sys.modules["google"] = google_mod
            sys.modules["google.generativeai"] = google_genai_mod

            openai_mod = types.ModuleType("openai")
            openai_mod.OpenAI = type("DummyOpenAI", (), {})
            sys.modules["openai"] = openai_mod

            try:
                from PIL import Image as _
            except ImportError:
                pil_mod = types.ModuleType("PIL")
                pil_image_mod = types.ModuleType("PIL.Image")
                pil_image_mod.open = lambda *_args, **_kwargs: None
                pil_mod.Image = pil_image_mod
                sys.modules["PIL"] = pil_mod
                sys.modules["PIL.Image"] = pil_image_mod

            generation_guide_mod = types.ModuleType("embodiedbench.planner.planner_config.generation_guide")
            generation_guide_mod.llm_generation_guide = {}
            generation_guide_mod.vlm_generation_guide = {}
            sys.modules["embodiedbench.planner.planner_config.generation_guide"] = generation_guide_mod

            generation_guide_manip_mod = types.ModuleType("embodiedbench.planner.planner_config.generation_guide_manip")
            generation_guide_manip_mod.llm_generation_guide_manip = {}
            generation_guide_manip_mod.vlm_generation_guide_manip = {}
            sys.modules["embodiedbench.planner.planner_config.generation_guide_manip"] = generation_guide_manip_mod

            planner_utils_mod = types.ModuleType("embodiedbench.planner.planner_utils")
            planner_utils_mod.convert_format_2claude = lambda message_history: message_history
            planner_utils_mod.convert_format_2gemini = lambda message_history: message_history
            planner_utils_mod.ActionPlan_1 = type("ActionPlan_1", (), {})
            planner_utils_mod.ActionPlan = type("ActionPlan", (), {})
            planner_utils_mod.ActionPlan_lang = type("ActionPlan_lang", (), {})
            planner_utils_mod.ActionPlan_1_manip = type("ActionPlan_1_manip", (), {})
            planner_utils_mod.ActionPlan_manip = type("ActionPlan_manip", (), {})
            planner_utils_mod.ActionPlan_lang_manip = type("ActionPlan_lang_manip", (), {})
            planner_utils_mod.fix_json = lambda text: text
            sys.modules["embodiedbench.planner.planner_utils"] = planner_utils_mod

            sys.modules.pop("lmdeploy", None)
            sys.meta_path.insert(0, BlockLmdeployImporter())
            module = importlib.import_module("embodiedbench.planner.remote_model")
            assert hasattr(module, "RemoteModel")
            assert "lmdeploy" not in sys.modules
            """
        )
        env = dict(os.environ)
        repo_root = os.path.dirname(os.path.dirname(__file__))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()

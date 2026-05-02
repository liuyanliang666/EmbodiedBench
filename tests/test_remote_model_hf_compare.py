import base64
import importlib
import io
import os
import sys
import types
import unittest


def _install_remote_model_stub_dependencies():
    anthropic_mod = types.ModuleType("anthropic")

    class DummyAnthropic:
        def __init__(self, *_args, **_kwargs):
            pass

    anthropic_mod.Anthropic = DummyAnthropic
    sys.modules["anthropic"] = anthropic_mod

    google_mod = types.ModuleType("google")
    google_genai_mod = types.ModuleType("google.generativeai")
    google_mod.generativeai = google_genai_mod
    sys.modules["google"] = google_mod
    sys.modules["google.generativeai"] = google_genai_mod

    openai_mod = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, *_args, **_kwargs):
            pass

    openai_mod.OpenAI = DummyOpenAI
    sys.modules["openai"] = openai_mod

    # lmdeploy stubs (for 'local' path)
    lmdeploy_mod = types.ModuleType("lmdeploy")

    class DummyResponse:
        def __init__(self, text="local-output"):
            self.text = text

    class DummyPipeline:
        def __call__(self, *_args, **_kwargs):
            return DummyResponse()

    class DummyGenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyPytorchEngineConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def dummy_pipeline(*_args, **_kwargs):
        return DummyPipeline()

    lmdeploy_mod.pipeline = dummy_pipeline
    lmdeploy_mod.GenerationConfig = DummyGenerationConfig
    lmdeploy_mod.PytorchEngineConfig = DummyPytorchEngineConfig
    sys.modules["lmdeploy"] = lmdeploy_mod

    class _FakeTensor:
        def __init__(self, data=None, shape=None, dtype=None):
            if shape is not None:
                self._shape = tuple(shape)
            elif data is not None:
                self._shape = (len(data),) if isinstance(data, (list, tuple)) else (1,)
            else:
                self._shape = (1,)
            self.dtype = dtype

        @property
        def shape(self):
            return self._shape

        def to(self, *_args, **_kwargs):
            return self

        def __getitem__(self, key):
            return self

    class _FakeDevice:
        def __init__(self, name="cpu"):
            self.type = name

        def __repr__(self):
            return f"device('{self.type}')"

    def _ones(*size, dtype=None):
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            size = tuple(size[0])
        return _FakeTensor(shape=size, dtype=dtype)

    def _empty(*size, dtype=None):
        return _FakeTensor(shape=size, dtype=dtype)

    # torch stub (for environments without GPU dependencies)
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        sys.modules["torch"] = torch_mod
    import torch  # noqa: E402 — now always available
    if not hasattr(torch, "Tensor"):
        torch.Tensor = _FakeTensor
    if not hasattr(torch, "device"):
        torch.device = _FakeDevice
    if not hasattr(torch, "ones"):
        torch.ones = _ones
    if not hasattr(torch, "empty"):
        torch.empty = _empty
    if not hasattr(torch, "float16"):
        torch.float16 = "float16"
    if not hasattr(torch, "long"):
        torch.long = "long"
    if not hasattr(torch, "inference_mode"):
        torch.inference_mode = lambda: type("ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()

    # transformers stubs (for 'hf_local' path)

    transformers_mod = types.ModuleType("transformers")

    class DummyProcessor:
        def apply_chat_template(self, messages, **kwargs):
            # Concatenate text items into a simple prompt
            parts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(item["text"])
                        elif item.get("type") == "image":
                            parts.append("<|image_pad|>")
            return " ".join(parts)

        def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
            seq_len = 10
            return {
                "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            }

        def batch_decode(self, token_ids, **kwargs):
            return ["hf-local-output"]

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class DummyModel:
        device = torch.device("cpu")

        def eval(self):
            return self

        def generate(self, **kwargs):
            prompt_len = kwargs.get("input_ids", torch.empty(1, 0)).shape[1]
            return torch.ones(1, prompt_len + 5, dtype=torch.long)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers_mod.AutoProcessor = DummyProcessor
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [101, 102, 103]

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers_mod.AutoTokenizer = DummyTokenizer
    transformers_mod.AutoModelForVision2Seq = DummyModel
    transformers_mod.AutoModelForCausalLM = DummyModel
    sys.modules["transformers"] = transformers_mod

    vllm_mod = types.ModuleType("vllm")

    class DummySamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyGenerateOutput:
        def __init__(self, text="vllm-direct-output"):
            self.outputs = [types.SimpleNamespace(text=text)]

    class DummyLLM:
        def __init__(self, *_args, **_kwargs):
            self.last_prompts = None
            self.last_sampling_params = None

        def generate(self, prompts, sampling_params=None, use_tqdm=False):
            self.last_prompts = prompts
            self.last_sampling_params = sampling_params
            return [DummyGenerateOutput()]

    vllm_mod.LLM = DummyLLM
    vllm_mod.SamplingParams = DummySamplingParams
    sys.modules["vllm"] = vllm_mod

    # PIL stub — only needed if PIL is not installed in the test env
    try:
        from PIL import Image as _  # noqa: F401
    except ImportError:
        pil_mod = types.ModuleType("PIL")
        pil_image_mod = types.ModuleType("PIL.Image")

        class _FakeImage:
            def __init__(self, size=(1, 1), mode="RGB"):
                self.size = size
                self.mode = mode

            def convert(self, mode):
                return _FakeImage(size=self.size, mode=mode)

        def _fake_open(*_args, **_kwargs):
            return _FakeImage()

        pil_image_mod.Image = _FakeImage
        pil_image_mod.open = _fake_open
        # Make "from PIL import Image" work and Image.open / instances behave
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


_install_remote_model_stub_dependencies()
sys.modules.pop("embodiedbench.planner.remote_model", None)
remote_model_module = importlib.import_module("embodiedbench.planner.remote_model")
RemoteModel = remote_model_module.RemoteModel


def _make_1x1_png_data_url():
    """Create a minimal 1x1 red PNG as a base64 data URL (no PIL needed)."""
    # Minimal valid 1x1 red PNG (hand-crafted)
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'  # PNG signature
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02'
        b'\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx'
        b'\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class RemoteModelLocalTests(unittest.TestCase):
    """Tests for the existing lmdeploy 'local' path."""

    def test_call_local_easyr1_format(self):
        model = RemoteModel("dummy-model", model_type="local", use_easyr1_format=True)
        out = model._call_local([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
        self.assertEqual(out, "local-output")

    def test_call_local_standard_format(self):
        model = RemoteModel("dummy-model", model_type="local", use_easyr1_format=False)
        out = model._call_local([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
        self.assertEqual(out, "local-output")


class ExtractImagesFromMessagesTests(unittest.TestCase):
    """Tests for _extract_images_from_messages."""

    def test_text_only_message(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        hf_msgs, images = RemoteModel._extract_images_from_messages(msgs)

        self.assertEqual(len(hf_msgs), 1)
        self.assertEqual(hf_msgs[0]["content"][0]["type"], "text")
        self.assertEqual(images, [])

    def test_string_content_passthrough(self):
        msgs = [{"role": "assistant", "content": "world"}]
        hf_msgs, images = RemoteModel._extract_images_from_messages(msgs)

        self.assertEqual(hf_msgs[0]["content"], "world")
        self.assertEqual(images, [])

    def test_image_url_extraction(self):
        data_url = _make_1x1_png_data_url()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "describe this"},
                ],
            }
        ]
        hf_msgs, images = RemoteModel._extract_images_from_messages(msgs)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].mode, "RGB")
        self.assertEqual(hf_msgs[0]["content"][0], {"type": "image"})
        self.assertEqual(hf_msgs[0]["content"][1]["type"], "text")

    def test_multiple_images(self):
        data_url = _make_1x1_png_data_url()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "second"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        hf_msgs, images = RemoteModel._extract_images_from_messages(msgs)

        self.assertEqual(len(images), 2)
        img_items = [i for i in hf_msgs[0]["content"] if i.get("type") == "image"]
        self.assertEqual(len(img_items), 2)


class RemoteModelHFLocalTests(unittest.TestCase):
    """Tests for the new 'hf_local' path."""

    def setUp(self):
        torch = sys.modules["torch"]
        if not hasattr(torch, "Tensor"):
            class _FakeTensor:
                def __init__(self, data=None, shape=None, dtype=None):
                    if shape is not None:
                        self._shape = tuple(shape)
                    elif data is not None:
                        self._shape = (len(data),) if isinstance(data, (list, tuple)) else (1,)
                    else:
                        self._shape = (1,)
                    self.dtype = dtype

                @property
                def shape(self):
                    return self._shape

                def to(self, *_args, **_kwargs):
                    return self

                def __getitem__(self, key):
                    return self

            torch.Tensor = _FakeTensor
        if not hasattr(torch, "ones"):
            def _ones(*size, dtype=None):
                if len(size) == 1 and isinstance(size[0], (list, tuple)):
                    size = tuple(size[0])
                return torch.Tensor(shape=size, dtype=dtype)

            torch.ones = _ones
        if not hasattr(torch, "empty"):
            def _empty(*size, dtype=None):
                if len(size) == 1 and isinstance(size[0], (list, tuple)):
                    size = tuple(size[0])
                return torch.Tensor(shape=size, dtype=dtype)

            torch.empty = _empty
        if not hasattr(torch, "inference_mode"):
            torch.inference_mode = lambda: type("ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()
        if not hasattr(torch, "float16"):
            torch.float16 = "float16"
        if not hasattr(torch, "long"):
            torch.long = "long"

    def test_build_qwen_prompt_interleaves_image_tokens(self):
        prompt = RemoteModel._build_qwen_prompt_text(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Task: put spoon.\n"},
                        {"type": "image"},
                        {"type": "text", "text": "\nChoose one action."},
                    ],
                }
            ]
        )
        self.assertEqual(
            prompt,
            "<|im_start|>user\n"
            "Task: put spoon.\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "\nChoose one action."
            "<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_hf_local_init(self):
        model = RemoteModel("dummy-model", model_type="hf_local", use_easyr1_format=True)
        self.assertTrue(hasattr(model, "processor"))
        self.assertEqual(model.model_type, "hf_local")

    def test_call_hf_local_text_only(self):
        model = RemoteModel("dummy-model", model_type="hf_local", use_easyr1_format=True)
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        out = model._call_hf_local(msgs)
        self.assertEqual(out, "hf-local-output")

    def test_call_hf_local_with_image(self):
        model = RemoteModel("dummy-model", model_type="hf_local", use_easyr1_format=True)
        data_url = _make_1x1_png_data_url()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        out = model._call_hf_local(msgs)
        self.assertEqual(out, "hf-local-output")

    def test_call_hf_local_qwen_does_not_require_apply_chat_template(self):
        model = RemoteModel("Qwen2.5-VL-7B-Instruct", model_type="hf_local", use_easyr1_format=True)
        captured = {}

        class CaptureProcessor:
            def apply_chat_template(self, *_args, **_kwargs):
                raise ImportError("jinja2>=3.1.0 required")

            def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
                import torch

                captured["text"] = text
                captured["images"] = images
                return {
                    "input_ids": torch.ones(1, 10, dtype=torch.long),
                    "attention_mask": torch.ones(1, 10, dtype=torch.long),
                }

            def batch_decode(self, token_ids, **kwargs):
                return ["hf-local-output"]

        model.processor = CaptureProcessor()
        data_url = _make_1x1_png_data_url()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task: open fridge.\n"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "\nOutput one action."},
                ],
            }
        ]

        out = model._call_hf_local(msgs)

        self.assertEqual(out, "hf-local-output")
        self.assertEqual(
            captured["text"],
            [
                "<|im_start|>user\n"
                "Task: open fridge.\n"
                "<|vision_start|><|image_pad|><|vision_end|>"
                "\nOutput one action."
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ],
        )
        self.assertEqual(len(captured["images"]), 1)

    def test_respond_routes_hf_local(self):
        model = RemoteModel("dummy-model", model_type="hf_local", use_easyr1_format=True)
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        out = model.respond(msgs)
        self.assertEqual(out, "hf-local-output")

    def test_call_hf_local_non_easyr1_applies_fix_json(self):
        model = RemoteModel("dummy-model", model_type="hf_local", use_easyr1_format=False)
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        # fix_json is stubbed as identity, so output passes through unchanged
        out = model._call_hf_local(msgs)
        self.assertEqual(out, "hf-local-output")


class RemoteModelVLLMDirectTests(unittest.TestCase):
    def test_respond_routes_vllm_direct(self):
        model = RemoteModel("Qwen2.5-VL-7B-Instruct", model_type="vllm_direct", use_easyr1_format=True)
        model._call_vllm_direct = lambda _msgs: "vllm-direct-output"
        model._call_qwen7b = lambda _msgs: "wrong-route"

        out = model.respond([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])

        self.assertEqual(out, "vllm-direct-output")

    def test_call_vllm_direct_builds_prompt_token_ids_and_multi_modal_data(self):
        model = RemoteModel("Qwen2.5-VL-7B-Instruct", model_type="vllm_direct", use_easyr1_format=True)

        captured = {}

        class CaptureTokenizer:
            def encode(self, text, add_special_tokens=False):
                captured["prompt_text"] = text
                captured["add_special_tokens"] = add_special_tokens
                return [11, 22, 33]

        class CaptureVLLMEngine:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                captured["prompts"] = prompts
                captured["sampling_params"] = sampling_params
                captured["use_tqdm"] = use_tqdm
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="vllm-direct-output")])]

        model.tokenizer = CaptureTokenizer()
        model.vllm_engine = CaptureVLLMEngine()

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task: pick up apple.\n"},
                    {"type": "image_url", "image_url": {"url": _make_1x1_png_data_url()}},
                    {"type": "text", "text": "\nOutput one action."},
                ],
            }
        ]

        out = model._call_vllm_direct(msgs)

        self.assertEqual(out, "vllm-direct-output")
        self.assertEqual(captured["prompt_text"], "<|im_start|>user\nTask: pick up apple.\n<|vision_start|><|image_pad|><|vision_end|>\nOutput one action.<|im_end|>\n<|im_start|>assistant\n")
        self.assertFalse(captured["add_special_tokens"])
        self.assertEqual(captured["prompts"][0]["prompt_token_ids"], [11, 22, 33])
        self.assertEqual(len(captured["prompts"][0]["multi_modal_data"]["image"]), 1)
        self.assertFalse(captured["use_tqdm"])

    def test_call_vllm_direct_accepts_raw_image_items(self):
        model = RemoteModel("Qwen2.5-VL-7B-Instruct", model_type="vllm_direct", use_easyr1_format=True)

        raw_image = object()
        captured = {}

        class CaptureTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [7, 8, 9]

        class CaptureVLLMEngine:
            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                captured["prompts"] = prompts
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="vllm-direct-output")])]

        model.tokenizer = CaptureTokenizer()
        model.vllm_engine = CaptureVLLMEngine()

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task: pick up apple.\n"},
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": "\nOutput one action."},
                ],
            }
        ]

        out = model._call_vllm_direct(msgs)

        self.assertEqual(out, "vllm-direct-output")
        self.assertIs(captured["prompts"][0]["multi_modal_data"]["image"][0], raw_image)


if __name__ == "__main__":
    unittest.main()

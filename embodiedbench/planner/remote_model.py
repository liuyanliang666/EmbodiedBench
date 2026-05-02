import json
import io
import sys
import os
import base64
import anthropic
import google.generativeai as genai
from openai import OpenAI
import typing_extensions as typing
import lmdeploy
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from PIL import Image
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_config.generation_guide_manip import llm_generation_guide_manip, vlm_generation_guide_manip
from embodiedbench.planner.planner_utils import convert_format_2claude, convert_format_2gemini, ActionPlan_1, ActionPlan, ActionPlan_lang, \
                                             ActionPlan_1_manip, ActionPlan_manip, ActionPlan_lang_manip, fix_json

temperature = 0
max_completion_tokens = 2048
remote_url = os.environ.get('remote_url')

_DEFAULT_SYSTEM_PROMPT_PREFIXES = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n",
)
_QWEN_VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

class RemoteModel:
    def __init__(
        self,
        model_name,
        model_type='remote',
        language_only=False,
        tp=1,
        task_type=None, # used to distinguish between manipulation and other environments
        use_easyr1_format=False
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.task_type = task_type
        self.use_easyr1_format = use_easyr1_format

        if self.model_type == 'local':
            backend_config = PytorchEngineConfig(session_len=12000, dtype='float16', tp=tp)
            self.model = pipeline(self.model_name, backend_config=backend_config)
        elif self.model_type == 'hf_local':
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name, torch_dtype=torch.float16, device_map="auto",
                )
            except ValueError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=torch.float16, device_map="auto",
                )
            self.model.eval()
        elif self.model_type == 'vllm_direct':
            from transformers import AutoProcessor, AutoTokenizer
            from vllm import LLM, SamplingParams

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.vllm_engine = LLM(self.model_name, tensor_parallel_size=tp)
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_completion_tokens,
            )
        else:
            if "claude" in self.model_name:
                self.model = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            elif "gemini" in self.model_name:
                self.model = OpenAI(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            elif "gpt" in self.model_name:
                self.model = OpenAI()
            elif 'qwen' in self.model_name:
                self.model = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif "Qwen2-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Qwen2.5-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "OpenGVLab/InternVL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "90b-vision-instruct" in self.model_name: # you can use fireworks to inference
                self.model = OpenAI(base_url='https://api.fireworks.ai/inference/v1',
                                    api_key=os.environ.get("firework_API_KEY"))
            else:
                try:
                    self.model = OpenAI(base_url = remote_url)
                except:
                    raise ValueError(f"Unsupported model name: {model_name}")


    def respond(self, message_history: list):
        if self.model_type == 'local':
            return self._call_local(message_history)
        elif self.model_type == 'hf_local':
            return self._call_hf_local(message_history)
        elif self.model_type == 'vllm_direct':
            return self._call_vllm_direct(message_history)
        else:
            if "claude" in self.model_name:
                return self._call_claude(message_history)
            elif "gemini" in self.model_name:
                return self._call_gemini(message_history)
            elif "gpt" in self.model_name:
                return self._call_gpt(message_history)
            elif 'qwen' in self.model_name:
                return self._call_gpt(message_history)
            elif "Qwen2-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Qwen2.5-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                return self._call_llama11b(message_history)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "90b-vision-instruct" in self.model_name:
                return self._call_llama90(message_history)
            elif "OpenGVLab/InternVL" in self.model_name:
                return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-38B" in self.model_name:
            #     return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-78B" in self.model_name:
            #     return self._call_intern38b(message_history)
            else:
                return self._call_openai_text(message_history)

    def _call_openai_text(self, message_history: list, model_name=None):
        response = self.model.chat.completions.create(
            model=model_name or self.model_name,
            messages=message_history,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        return response.choices[0].message.content

    def _call_local(self, message_history: list):
        if self.use_easyr1_format:
            response = self.model(
                message_history,
                gen_config=GenerationConfig(
                    temperature=temperature,
                    max_new_tokens=max_completion_tokens,
                )
            )
            return response.text

        if self.task_type == 'manip':
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": llm_generation_guide_manip if self.language_only else vlm_generation_guide_manip
                }
            }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": llm_generation_guide if self.language_only else vlm_generation_guide
                }
            }
        response = self.model(
            message_history,
            gen_config=GenerationConfig(
                temperature=temperature,
                response_format=response_format,
                max_new_tokens=max_completion_tokens,
            )
        )
        out = response.text
        out = fix_json(out)
        return out

    @staticmethod
    def _extract_images_from_messages(message_history):
        """Convert OpenAI-format messages to a processor-friendly message format.

        Replaces ``image_url`` items (with base64 data URLs) with
        ``{"type": "image"}`` and extracts the corresponding PIL images.

        Returns:
            hf_messages: transformed message list for prompt serialization
            pil_images:  list of PIL.Image in order of appearance
        """
        pil_images = []
        hf_messages = []
        for msg in message_history:
            new_msg = {"role": msg["role"]}
            content = msg.get("content")
            if isinstance(content, str):
                new_msg["content"] = content
                hf_messages.append(new_msg)
                continue
            new_content = []
            for item in (content or []):
                item_type = item.get("type")
                if item_type == "image_url":
                    url = item["image_url"]["url"]
                    _, b64_data = url.split(",", 1)
                    image_bytes = base64.b64decode(b64_data)
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    pil_images.append(pil_img)
                    new_content.append({"type": "image"})
                elif item_type == "image":
                    pil_images.append(item.get("image"))
                    new_content.append({"type": "image"})
                else:
                    new_content.append(item)
            new_msg["content"] = new_content
            hf_messages.append(new_msg)
        return hf_messages, pil_images

    @staticmethod
    def _is_qwen_model_name(model_name):
        lowered = (model_name or "").lower()
        return "qwen" in lowered

    @classmethod
    def _serialize_qwen_content(cls, content):
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if not isinstance(content, list):
            raise TypeError(f"Unsupported Qwen message content type: {type(content)!r}")

        parts = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type in {"image", "image_url"}:
                parts.append(_QWEN_VISION_PLACEHOLDER)
            else:
                raise ValueError(f"Unsupported Qwen content item type: {item_type}")
        return "".join(parts)

    @classmethod
    def _build_qwen_prompt_text(cls, messages, add_generation_prompt=True):
        parts = []
        for message in messages:
            role = message.get("role", "user")
            parts.append(f"<|im_start|>{role}\n")
            parts.append(cls._serialize_qwen_content(message.get("content")))
            parts.append("<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _render_hf_local_prompt(self, hf_messages):
        if self._is_qwen_model_name(self.model_name):
            return self._build_qwen_prompt_text(hf_messages, add_generation_prompt=True)

        prompt_text = self.processor.apply_chat_template(
            hf_messages, add_generation_prompt=True, tokenize=False,
        )

        # Strip default Qwen system prompt injected by apply_chat_template
        # (matches EasyR1 _strip_leading_default_system_prompt behaviour)
        if not hf_messages or hf_messages[0].get("role") != "system":
            for prefix in _DEFAULT_SYSTEM_PROMPT_PREFIXES:
                if prompt_text.startswith(prefix):
                    prompt_text = prompt_text[len(prefix):]
                    break
        return prompt_text

    def _call_hf_local(self, message_history: list):
        """Local inference using HF AutoModelForCausalLM + AutoProcessor.

        Tokenisation matches EasyR1 / DeepEyes training exactly.
        """
        import torch

        hf_messages, pil_images = self._extract_images_from_messages(message_history)
        prompt_text = self._render_hf_local_prompt(hf_messages)

        if pil_images:
            inputs = self.processor(
                text=[prompt_text], images=pil_images, return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[prompt_text], return_tensors="pt",
            )

        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_completion_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
            )

        generated_ids = output_ids[:, prompt_len:]
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if not self.use_easyr1_format:
            out = fix_json(out)
        return out

    def _call_vllm_direct(self, message_history: list):
        """Direct vLLM inference using prompt_token_ids + multi_modal_data."""
        hf_messages, pil_images = self._extract_images_from_messages(message_history)
        prompt_text = self._render_hf_local_prompt(hf_messages)
        prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        vllm_input = {"prompt_token_ids": prompt_token_ids}
        if pil_images:
            vllm_input["multi_modal_data"] = {"image": pil_images}

        outputs = self.vllm_engine.generate(
            [vllm_input],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        out = outputs[0].outputs[0].text

        if not self.use_easyr1_format:
            out = fix_json(out)
        return out

    def _call_claude(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2claude(message_history)

        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            messages=message_history
        )

        return response.content[0].text 

    def _call_gemini(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.use_easyr1_format:
            return self._call_openai_text(message_history)

        if self.task_type == 'manip':
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                response_format= ActionPlan_lang_manip if self.language_only else ActionPlan_manip,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        else:
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                response_format= ActionPlan_lang if self.language_only else ActionPlan,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        tokens = response.usage.prompt_tokens

        return str(response.choices[0].message.parsed.model_dump_json())

    def _call_gpt(self, message_history: list):
        if self.use_easyr1_format:
            return self._call_openai_text(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content

        return out
    
    def _call_qwen7b(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.use_easyr1_format:
            return self._call_openai_text(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        return out
    
    def _call_llama90(self, message_history: list):
        if self.use_easyr1_format:
            return self._call_openai_text(message_history, model_name="accounts/fireworks/models/llama-v3p2-90b-vision-instruct")

        if self.task_type == "manip":
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={"type": "json_object", "schema": ActionPlan_1_manip.model_json_schema()},
                temperature = temperature
            )
            out = response.choices[0].message.content
            
        else:
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={"type": "json_object", "schema": ActionPlan_1.model_json_schema()},
                temperature = temperature
            )
            out = response.choices[0].message.content
        return out
    
    def _call_llama11b(self, message_history):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.use_easyr1_format:
            return self._call_openai_text(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        return out
    

    def _call_qwen72b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.use_easyr1_format:
            return self._call_openai_text(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))
        
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    
    def _call_intern38b(self, message_history):

        # if not self.language_only:
        #     message_history = convert_format_2gemini(message_history)

        if self.use_easyr1_format:
            return self._call_openai_text(message_history)

        # no use, lmdeploy use support json schema only if it is pytorch-backended
        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out



if __name__ == "__main__":

    model = RemoteModel(
        'Qwen/Qwen2-VL-72B-Instruct', #'meta-llama/Llama-3.2-11B-Vision-Instruct',
        True #False
    )#'claude-3-5-sonnet-20241022, Qwen/Qwen2-VL-72B-Instruct, meta-llama/Llama-3.2-11B-Vision-Instruct


    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

    base64_image = encode_image("../../evaluator/midlevel/output.png")
        
    messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                }
            },
            {
                "type": "text",
                "text":f"What do you think for this picture?? {template}?"
            },
            ],
        }
    ]

    response = model.respond(messages)
    print(response)

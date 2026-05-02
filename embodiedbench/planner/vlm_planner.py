import torch
import re
import os
import time
import numpy as np
import cv2
import json
import traceback
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url, template, template_lang, fix_json
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger

class VLMPlanner():
    def __init__(self, model_name, model_type, actions, system_prompt, examples, n_shot=0, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1,
                memory_compression=False, segment_len=1, enable_point_actions=False, use_easyr1_format=False,
                use_topdown_prompt=False, kwargs={}):
        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.set_actions(actions)
        self.model_type = model_type
        self.use_topdown_prompt = bool(use_topdown_prompt)
        self.use_easyr1_format = bool(use_easyr1_format or self.use_topdown_prompt)
        self.kwargs = dict(kwargs)
        self.action_key = self.kwargs.pop('action_key', 'action_id')
        self.image_resolution = int(self.kwargs.pop('image_resolution', 600))
        self.topdown_obs_key = str(self.kwargs.pop('topdown_obs_key', 'topdown_rgb'))
        self.prompt_template_path = self.kwargs.pop(
            'prompt_template_path',
            os.path.join(os.path.dirname(__file__), 'prompt_templates', 'alfred.jinja')
        )
        # The single alfred.jinja template now branches on `images | length` to
        # cover both first-person-only and first-person + topdown layouts. The
        # old separate topdown path is kept as a no-op kwarg for back-compat.
        self.kwargs.pop('topdown_prompt_template_path', None)
        self._prompt_template = None
        self._jinja_env = None
        self.easyr1_variant = str(self.kwargs.pop('easyr1_variant', 'default')).strip().lower()
        self.easyr1_navigation_only = (self.easyr1_variant == 'navigation')
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(
                model_name,
                model_type,
                language_only,
                tp=tp,
                use_easyr1_format=self.use_easyr1_format,
            )

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.memory_compression = bool(memory_compression)
        try:
            self.segment_len = max(1, int(segment_len))
        except (TypeError, ValueError):
            self.segment_len = 1
        self.enable_point_actions = enable_point_actions
        self.reset()
    
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str


    def _build_standard_initial_prompt(self, user_instruction):
        if self.n_shot >= 1:
            prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i, x in enumerate(self.examples[:self.n_shot])]))
        else:
            prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

        prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
        if self.language_only:
            prompt += f" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
        else:
            prompt += f" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
        return prompt

    def _build_initial_prompt(self, user_instruction):
        if self.use_easyr1_format:
            return self._build_easyr1_prompt(user_instruction, [])
        return self._build_standard_initial_prompt(user_instruction)

    def _extract_observation_frame(self, observation):
        if observation is None:
            return None
        if isinstance(observation, dict):
            return observation.get(self.obs_key)
        return observation

    def _observations_are_unchanged(self, previous_obs, current_obs):
        previous_frame = self._extract_observation_frame(previous_obs)
        current_frame = self._extract_observation_frame(current_obs)
        if previous_frame is None or current_frame is None:
            return False
        if hasattr(previous_frame, 'shape') and hasattr(current_frame, 'shape'):
            if previous_frame.shape != current_frame.shape:
                return False
            return bool(np.array_equal(previous_frame, current_frame))
        return previous_frame == current_frame

    def _rebuild_episode_memory(self, previous_obs, current_obs, info=None):
        if self._observations_are_unchanged(previous_obs, current_obs):
            self.episode_memory = (
                "The previous action was invalid and did not change the scene. "
                "Before trying again, consider these likely causes: "
                "the target is not visible from the current viewpoint; "
                "the target is too far away or out of reach; "
                "the camera is too close to interact accurately; "
                "a receptacle or object that must be opened is still closed; "
                "the path or interaction is blocked; "
                "a required object is not in hand; "
                "the agent is already holding the wrong object; "
                "the selected interaction point is likely inaccurate; "
                "or the selected interaction, viewpoint, or prerequisite was likely incorrect."
            )
        else:
            self.episode_memory = ""
        # Update inventory state from latest feedback
        if info is not None:
            inventory_line = self._extract_inventory_line([[None, info.get('env_feedback', '')]])
            self.episode_inventory = inventory_line or ""

    def _build_memory_prompt(self, user_instruction):
        prompt = self._build_initial_prompt(user_instruction)
        if self.episode_inventory:
            prompt += f"\n\n{self.episode_inventory}"
        if self.episode_memory:
            prompt += f"\n\n{self.episode_memory}"
        return prompt

    def process_prompt(self, user_instruction, prev_act_feedback=[]):
        if self.memory_compression:
            return self._build_memory_prompt(user_instruction.rstrip('.'))
        if self.use_easyr1_format:
            return self._build_easyr1_prompt(user_instruction, prev_act_feedback)

        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            prompt = self._build_standard_initial_prompt(user_instruction)

        elif self.chat_history:
            prompt = f'The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                action_desc = self._format_action_feedback(action_feedback[0])
                if self.use_feedback:
                    prompt += '\nStep {}, action {}, env feedback: {}'.format(i, action_desc, action_feedback[1])
                else:
                    prompt += '\nStep {}, action {}'.format(i, action_desc)

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i, x in enumerate(self.examples[:self.n_shot])]))
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                action_desc = self._format_action_feedback(action_feedback[0])
                if self.use_feedback:
                    prompt += '\nStep {}, action {}, env feedback: {}'.format(i, action_desc, action_feedback[1])
                else:
                    prompt += '\nStep {}, action {}'.format(i, action_desc)

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        return prompt
    @staticmethod
    def _extract_inventory_line(prev_act_feedback):
        """Extract the most recent 'Currently holding' status from feedback history."""
        held = VLMPlanner._extract_held_object_name(prev_act_feedback)
        if held is None:
            return None
        if held.lower() == 'nothing':
            return 'You are currently holding nothing in your hand.'
        return f'You are currently holding: {held}.'

    @staticmethod
    def _extract_held_object_name(prev_act_feedback):
        """Return the held-object name from the most recent env feedback, or None.

        Mirrors the field consumed by alfred.jinja's `held_object` placeholder
        (see EasyR1 generate_summaries_v4.py). Returns 'nothing' when the agent
        explicitly holds nothing, the bare object name otherwise.
        """
        if not prev_act_feedback:
            return None
        last_feedback = prev_act_feedback[-1]
        feedback_text = last_feedback[1] if isinstance(last_feedback, (list, tuple)) and len(last_feedback) >= 2 else ''
        if not isinstance(feedback_text, str):
            return None
        m = re.search(r'Currently holding: ([^.]+)\.', feedback_text)
        if m:
            held = m.group(1).strip()
            # Training data sources held_object from ALFRED PDDL discrete_action
            # args, which are lowercase (e.g. "apple", "floorlamp"). The env
            # feedback uses THOR's PascalCase objectType ("Apple", "FloorLamp").
            # Lowercase to match the training distribution.
            if held.lower() == 'nothing':
                return 'nothing'
            return held.lower()
        return None

    @staticmethod
    def _extract_summary_block(output_text):
        """Pull the full <summary>...</summary> block from a model response.

        Used to feed `prev_summary` into the next step's prompt. The wrapping
        tags are preserved so the next prompt sees the same structure the
        model was trained to emit.
        """
        if not isinstance(output_text, str):
            return ""
        match = re.search(r'<summary>\s*(.*?)\s*</summary>', output_text, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            return ""
        inner = match.group(1).strip()
        if not inner:
            return ""
        return f"<summary>\n{inner}\n</summary>"

    @staticmethod
    def _format_action_description(action_dict):
        """Render an action dict as the human-readable string the v4 trainer used.

        Matches generate_summaries_v4.get_action_description so that the
        `last_action` field at eval time aligns with the training distribution:
        list parameters → "ActionType at [x, y]"; string parameters →
        "ActionType direction"; missing parameter → "ActionType".
        """
        if not isinstance(action_dict, dict):
            return ""
        action_type = action_dict.get('action_type', '')
        if not action_type:
            return ""
        parameter = action_dict.get('parameter', action_dict.get('action_param'))
        if isinstance(parameter, list):
            return f"{action_type} at {parameter}"
        if isinstance(parameter, str):
            return f"{action_type} {parameter}"
        return str(action_type)

    @classmethod
    def _last_action_from_output(cls, output_text):
        """Parse the model's <answer> JSON and format it as last_action text."""
        if not isinstance(output_text, str):
            return ""
        answer_text = cls._extract_answer_block(output_text)
        answer_text = cls._strip_code_fence(answer_text)
        try:
            parsed = json.loads(answer_text)
        except (json.JSONDecodeError, TypeError):
            return ""
        if isinstance(parsed, list):
            if not parsed:
                return ""
            parsed = parsed[0]
        return cls._format_action_description(parsed)

    @staticmethod
    def _stuck_hint_text():
        """Hint appended when consecutive observations are pixel-identical."""
        return (
            "The previous action was invalid and did not change the scene. "
            "Before trying again, consider these likely causes: "
            "the target is not visible from the current viewpoint; "
            "the target is too far away or out of reach; "
            "the camera is too close to interact accurately; "
            "a receptacle or object that must be opened is still closed; "
            "the path or interaction is blocked; "
            "a required object is not in hand; "
            "the agent is already holding the wrong object; "
            "the selected interaction point is likely inaccurate; "
            "or the selected interaction, viewpoint, or prerequisite was likely incorrect."
        )

    def _render_alfred_template(self, template_text, context):
        """Render alfred.jinja with the given context (jinja2-evaluated)."""
        if self._jinja_env is None:
            from jinja2 import Environment

            self._jinja_env = Environment(
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=False,
                keep_trailing_newline=False,
            )
        return self._jinja_env.from_string(template_text).render(**context)

    def _build_easyr1_prompt(self, user_instruction, prev_act_feedback=[]):
        if self.easyr1_navigation_only:
            if self.use_topdown_prompt:
                return self._build_navigation_topdown_easyr1_prompt(user_instruction)
            return self._build_navigation_easyr1_prompt(user_instruction)
        return self._build_alfred_easyr1_prompt(
            user_instruction,
            prev_act_feedback,
            with_topdown=self.use_topdown_prompt,
        )

    def _load_prompt_template(self):
        if self._prompt_template is None:
            with open(self.prompt_template_path, 'r', encoding='utf-8') as handle:
                self._prompt_template = handle.read()
        return self._prompt_template

    def _build_alfred_easyr1_prompt(self, user_instruction, prev_act_feedback, with_topdown):
        """Render alfred.jinja with the EasyR1 v4-summary recursion fields.

        Each step receives `held_object`, `last_action`, and `prev_summary` so
        the model can recurse on its own prior summary, mirroring how the
        EasyR1 trainer composed its inputs (see generate_summaries_v4.py).
        """
        task = user_instruction.strip().rstrip('.')
        template_text = self._load_prompt_template()
        # Match the exact `Task:` line built by EasyR1's
        # convert_alfred_to_dataset.build_prompt (no trailing period).
        if with_topdown:
            content = '\n'.join([
                f'Task: {task}',
                'Current first-person view:',
                '<image>',
                'Reconstructed top-down occupancy map from the trajectory observed so far:',
                '<image>',
            ])
            images_marker = [None, None]
        else:
            content = '\n'.join([f'Task: {task}', '<image>'])
            images_marker = [None]

        held_object = self._extract_held_object_name(prev_act_feedback) or 'nothing'
        last_action = self.prev_action_desc or 'none (first step)'
        prev_summary = self.prev_summary or '(none, first step)'

        prompt = self._render_alfred_template(
            template_text,
            {
                'content': content,
                'held_object': held_object,
                'last_action': last_action,
                'prev_summary': prev_summary,
                'images': images_marker,
            },
        )

        if self.frames_unchanged_hint:
            prompt = f"{prompt}\n\n{self.frames_unchanged_hint}"

        return prompt

    def _build_topdown_easyr1_prompt(self, user_instruction, prev_act_feedback=[]):
        # Retained for back-compat; delegate to the unified alfred renderer.
        return self._build_alfred_easyr1_prompt(
            user_instruction, prev_act_feedback, with_topdown=True
        )

    def _build_navigation_easyr1_prompt(self, user_instruction):
        task = user_instruction.strip().rstrip('.')
        lines = [
            'You are a household navigation robot. Given the current first-person photo and the task below, your goal is to predict the next optimal discrete navigation action.',
            f"Task: {task}",
            '<image>',
            'You need to use your prior knowledge about household layouts, target visibility, obstacles, and egocentric motion to move closer to the target object.',
            '',
            'Your action must be chosen from the following navigation-only list and formatted as a JSON array containing a single object. Each object must contain an "action_type" and a "parameter".',
            'You must output exactly ONE next action per turn (array length must be 1). Do not output multiple actions, plans, or alternative candidates.',
            '',
            'Navigation Actions:',
            "For these actions, the 'parameter' is a string indicating the direction.",
            '- Moves the robot forward by 25cm: [{"action_type": "Move", "parameter": "forward"}]',
            '- Moves the robot backward by 25cm: [{"action_type": "Move", "parameter": "backward"}]',
            '- Rotates the robot perspective 90 degrees to the right: [{"action_type": "Rotate", "parameter": "right"}]',
            '- Rotates the robot perspective 90 degrees to the left: [{"action_type": "Rotate", "parameter": "left"}]',
            '- Tilts the camera up by 15 degrees: [{"action_type": "Look", "parameter": "up"}]',
            '- Tilts the camera down by 15 degrees: [{"action_type": "Look", "parameter": "down"}]',
            '',
            'Output the thinking process in <think> </think> tags, and the final JSON answer in <answer> </answer> tags as follows:',
            'In <answer>, output only one JSON action object wrapped in a one-element array.',
            '<think> ... </think> <answer> answer here </answer>',
            '',
            'Examples:',
            '',
            '<think> The target object is not visible in the current frame. I should rotate right to explore the room and search for it. </think> <answer> [{"action_type": "Rotate", "parameter": "right"}] </answer>',
            '',
            '<think> I can already see the target object directly ahead, but it is still a few steps away. The best next action is to move forward. </think> <answer> [{"action_type": "Move", "parameter": "forward"}] </answer>',
            '',
            '<think> I am too close to a wall and need a slightly wider view before continuing toward the target. I should move backward first. </think> <answer> [{"action_type": "Move", "parameter": "backward"}] </answer>',
            '',
            '<think> The target seems lower than my current viewpoint and may be hidden by the countertop edge. I should tilt the camera down by 15 degrees. </think> <answer> [{"action_type": "Look", "parameter": "down"}] </answer>',
            '',
            '<think> The target is likely on a high shelf and not visible from the current camera pitch. I should tilt the camera up by 15 degrees. </think> <answer> [{"action_type": "Look", "parameter": "up"}] </answer>',
        ]
        return "\n".join(lines)

    def _build_navigation_topdown_easyr1_prompt(self, user_instruction):
        task = user_instruction.strip().rstrip('.')
        lines = [
            'You are a household navigation robot. Given the current first-person photo, the reconstructed top-down occupancy map, and the task below, your goal is to predict the next optimal discrete navigation action.',
            f"Task: {task}.",
            'Current first-person view:',
            '<image>',
            'Reconstructed top-down occupancy map from the trajectory observed so far:',
            '<image>',
            'You need to use your prior knowledge about household layouts, target visibility, obstacles, and egocentric motion to move closer to the target object.',
            '',
            'The first image is the current first-person photo. The second image is a reconstructed top-down occupancy map accumulated from the trajectory observed so far.',
            "In the top-down map, the blue arrow marks the robot's current location, and the direction the blue arrow points is the robot's current facing direction.",
            'In the top-down map, white regions indicate occupied obstacles, and black regions indicate non-occupied open space that is more likely to be traversable.',
            'Use the top-down map to reason about explored free space, room layout, obstacles, and navigation progress. Use the first-person photo to decide what is currently visible.',
            '',
            'Your action must be chosen from the following navigation-only list and formatted as a JSON array containing a single object. Each object must contain an "action_type" and a "parameter".',
            'You must output exactly ONE next action per turn (array length must be 1). Do not output multiple actions, plans, or alternative candidates.',
            '',
            'Navigation Actions:',
            "For these actions, the 'parameter' is a string indicating the direction.",
            '- Moves the robot forward by 25cm: [{"action_type": "Move", "parameter": "forward"}]',
            '- Moves the robot backward by 25cm: [{"action_type": "Move", "parameter": "backward"}]',
            '- Rotates the robot perspective 90 degrees to the right: [{"action_type": "Rotate", "parameter": "right"}]',
            '- Rotates the robot perspective 90 degrees to the left: [{"action_type": "Rotate", "parameter": "left"}]',
            '- Tilts the camera up by 15 degrees: [{"action_type": "Look", "parameter": "up"}]',
            '- Tilts the camera down by 15 degrees: [{"action_type": "Look", "parameter": "down"}]',
            '',
            'Output the thinking process in <think> </think> tags, and the final JSON answer in <answer> </answer> tags as follows:',
            'In <answer>, output only one JSON action object wrapped in a one-element array.',
            '<think> ... </think> <answer> answer here </answer>',
            '',
            'Examples:',
            '',
            '<think> The target object is not visible in the current first-person image. The top-down map shows unexplored free space on the robot\'s right side, so rotating right is the most useful next navigation action. </think> <answer> [{"action_type": "Rotate", "parameter": "right"}] </answer>',
            '',
            '<think> I can already see the target object directly ahead, and the top-down map shows free space in front of the agent. The best next action is to move forward. </think> <answer> [{"action_type": "Move", "parameter": "forward"}] </answer>',
            '',
            '<think> I am too close to a wall and need a slightly wider view before continuing toward the target. I should move backward first. </think> <answer> [{"action_type": "Move", "parameter": "backward"}] </answer>',
            '',
            '<think> The target seems lower than my current viewpoint and may be hidden by the countertop edge. I should tilt the camera down by 15 degrees. </think> <answer> [{"action_type": "Look", "parameter": "down"}] </answer>',
            '',
            '<think> The target is likely on a high shelf and not visible from the current camera pitch. I should tilt the camera up by 15 degrees. </think> <answer> [{"action_type": "Look", "parameter": "up"}] </answer>',
        ]
        return "\n".join(lines)
    _EASYR1_NAV_ACTIONS = {
        'MoveAhead': ('Move', 'forward'),
        'MoveBack': ('Move', 'backward'),
        'MoveRight': ('Move', 'right'),
        'MoveLeft': ('Move', 'left'),
        'RotateRight': ('Rotate', 'right'),
        'RotateLeft': ('Rotate', 'left'),
        'LookUp_15': ('Look', 'up'),
        'LookDown_15': ('Look', 'down'),
        'LookUp': ('Look', 'up'),
        'LookDown': ('Look', 'down'),
        'MoveAhead_25': ('Move', 'forward'),
        'MoveBack_25': ('Move', 'backward'),
        'MoveRight_25': ('Move', 'right'),
        'MoveLeft_25': ('Move', 'left'),
        'RotateRight_90': ('Rotate', 'right'),
        'RotateLeft_90': ('Rotate', 'left'),
    }
    _EASYR1_INTER_ACTIONS = {
        'pickup_by_point': 'PickupObject',
        'pick_by_point': 'PickupObject',
        'pickup_at_point': 'PickupObject',
        'pickup_point': 'PickupObject',
        'put_by_point': 'PutObject',
        'put_at_point': 'PutObject',
        'open_by_point': 'OpenObject',
        'close_by_point': 'CloseObject',
        'toggleon_by_point': 'ToggleObjectOn',
        'toggleoff_by_point': 'ToggleObjectOff',
        'slice_by_point': 'SliceObject',
        'pickupobject': 'PickupObject',
        'putobject': 'PutObject',
        'openobject': 'OpenObject',
        'closeobject': 'CloseObject',
        'toggleobjecton': 'ToggleObjectOn',
        'toggleobjectoff': 'ToggleObjectOff',
        'sliceobject': 'SliceObject',
    }
    _EASYR1_JSON_INTER_MAP = {
        'pickupobject': 'pickup_by_point',
        'putobject': 'put_by_point',
        'openobject': 'open_by_point',
        'closeobject': 'close_by_point',
        'toggleobjecton': 'toggleon_by_point',
        'toggleobjectoff': 'toggleoff_by_point',
        'sliceobject': 'slice_by_point',
    }

    _EASYR1_JSON_NAV_MAP = {
        ('move', 'forward'): 'MoveAhead',
        ('move', 'backward'): 'MoveBack',
        ('rotate', 'right'): 'RotateRight',
        ('rotate', 'left'): 'RotateLeft',
        ('look', 'up'): 'LookUp_15',
        ('look', 'down'): 'LookDown_15',
    }

    def _format_action_feedback(self, action_ref):
        if isinstance(action_ref, int):
            if 0 <= action_ref < len(self.actions):
                if self.use_easyr1_format:
                    formatted = self._format_action_feedback(self.actions[action_ref])
                    if formatted != self.actions[action_ref]:
                        return formatted
                return f"id {action_ref} ({self.actions[action_ref]})"
            return str(action_ref)
        if self.use_easyr1_format:
            formatted = self._format_easyr1_action(action_ref)
            if formatted is not None:
                return formatted
        return str(action_ref)

    @staticmethod
    def _strip_code_fence(text):
        match = re.match(r'^```(?:json)?\s*(.*?)\s*```$', text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def _normalize_easyr1_point(point):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except (TypeError, ValueError):
            return None
        return [x, y]

    def _to_easyr1_answer(self, action_type, parameter):
        return json.dumps([{"action_type": action_type, "parameter": parameter}], ensure_ascii=False)

    def _format_easyr1_action(self, action_ref):
        if isinstance(action_ref, dict):
            if 'action_type' in action_ref:
                parameter = action_ref.get('parameter', action_ref.get('action_param'))
                if self.easyr1_navigation_only:
                    action_type = str(action_ref['action_type']).strip().lower()
                    if action_type == 'move' and str(parameter).strip().lower() in ('right', 'left'):
                        return None
                    if action_type not in ('move', 'rotate', 'look'):
                        return None
                return self._to_easyr1_answer(action_ref['action_type'], parameter)

            action_name = str(action_ref.get('action', '')).strip()
            if self.easyr1_navigation_only and action_name in self._EASYR1_INTER_ACTIONS:
                return None
            point = self._normalize_easyr1_point(action_ref.get('point'))
            if action_name in self._EASYR1_INTER_ACTIONS and point is not None:
                return self._to_easyr1_answer(self._EASYR1_INTER_ACTIONS[action_name], point)
            return None

        if not isinstance(action_ref, str):
            return None

        stripped_action = action_ref.strip()
        if stripped_action in self._EASYR1_NAV_ACTIONS:
            action_type, parameter = self._EASYR1_NAV_ACTIONS[stripped_action]
            if self.easyr1_navigation_only and action_type == 'Move' and parameter in ('right', 'left'):
                return None
            return self._to_easyr1_answer(action_type, parameter)

        normalized = stripped_action.rstrip('.').lower()
        text_nav_map = {
            'move forward by 0.25': ('Move', 'forward'),
            'move backward by 0.25': ('Move', 'backward'),
            'move rightward by 0.25': ('Move', 'right'),
            'move leftward by 0.25': ('Move', 'left'),
            'rotate to the right by 90 degrees': ('Rotate', 'right'),
            'rotate to the left by 90 degrees': ('Rotate', 'left'),
            'tilt the camera upward by 30 degrees': ('Look', 'up'),
            'tilt the camera downward by 30 degrees': ('Look', 'down'),
        }
        if normalized in text_nav_map:
            action_type, parameter = text_nav_map[normalized]
            if self.easyr1_navigation_only and action_type == 'Move' and parameter in ('right', 'left'):
                return None
            return self._to_easyr1_answer(action_type, parameter)

        try:
            parsed = json.loads(stripped_action)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if isinstance(parsed, list) and len(parsed) >= 1:
            parsed = parsed[0]
        if isinstance(parsed, dict):
            if 'action_type' in parsed:
                parameter = parsed.get('parameter', parsed.get('action_param'))
                if self.easyr1_navigation_only:
                    action_type = str(parsed['action_type']).strip().lower()
                    if action_type == 'move' and str(parameter).strip().lower() in ('right', 'left'):
                        return None
                    if action_type not in ('move', 'rotate', 'look'):
                        return None
                return self._to_easyr1_answer(parsed['action_type'], parameter)

            action_name = str(parsed.get('action', '')).strip()
            if self.easyr1_navigation_only and action_name in self._EASYR1_INTER_ACTIONS:
                return None
            point = self._normalize_easyr1_point(parsed.get('point'))
            if action_name in self._EASYR1_INTER_ACTIONS and point is not None:
                return self._to_easyr1_answer(self._EASYR1_INTER_ACTIONS[action_name], point)

        return None

    @staticmethod
    def _summarize_easyr1_feedback_status(action_feedback):
        if isinstance(action_feedback, (list, tuple)) and len(action_feedback) >= 3:
            success_flag = action_feedback[2]
            if success_flag is not None:
                try:
                    return "success" if float(success_flag) > 0 else "invalid"
                except (TypeError, ValueError):
                    pass

        if isinstance(action_feedback, (list, tuple)) and len(action_feedback) >= 2:
            feedback_text = action_feedback[1]
        else:
            feedback_text = action_feedback

        if isinstance(feedback_text, str):
            normalized = feedback_text.strip().lower()
            if 'invalid' in normalized or 'failed' in normalized:
                return "invalid"
            if 'success' in normalized or 'executed successfully' in normalized:
                return "success"
        return "unknown"

    def _parse_point_action(self, action_step):
        if not isinstance(action_step, dict):
            return None

        action_name = str(action_step.get('action_name', '')).strip()
        lower_name = action_name.lower()

        explicit_action = str(action_step.get('action', '')).strip().lower()
        explicit_actions = {
            'pickup_by_point',
            'pick_by_point',
            'pickup_at_point',
            'pickup_point',
            'put_by_point',
            'put_at_point',
            'open_by_point',
            'close_by_point',
            'toggleon_by_point',
            'toggleoff_by_point',
            'slice_by_point',
        }
        if explicit_action in explicit_actions:
            point = action_step.get('point') or action_step.get('pixel') or action_step.get('click_point')
            if isinstance(point, (list, tuple)) and len(point) == 2:
                try:
                    x = int(round(float(point[0])))
                    y = int(round(float(point[1])))
                    return {"action": explicit_action, "point": [x, y]}
                except (TypeError, ValueError):
                    return None

        # Preferred explicit format: {"action_name": "...", "point": [x, y]}
        point = action_step.get('point') or action_step.get('pixel') or action_step.get('click_point')
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
                if 'pick' in lower_name:
                    return {"action": "pickup_by_point", "point": [x, y]}
                if 'put down' in lower_name or 'put' in lower_name or 'place' in lower_name:
                    return {"action": "put_by_point", "point": [x, y]}
                if 'turn on' in lower_name or 'toggle on' in lower_name:
                    return {"action": "toggleon_by_point", "point": [x, y]}
                if 'turn off' in lower_name or 'toggle off' in lower_name:
                    return {"action": "toggleoff_by_point", "point": [x, y]}
                if 'open' in lower_name:
                    return {"action": "open_by_point", "point": [x, y]}
                if 'close' in lower_name:
                    return {"action": "close_by_point", "point": [x, y]}
                if 'slice' in lower_name:
                    return {"action": "slice_by_point", "point": [x, y]}
            except (TypeError, ValueError):
                pass

        if not any(k in lower_name for k in ('pick', 'put', 'place', 'open', 'close', 'turn on', 'turn off', 'toggle on', 'toggle off', 'slice')):
            return None
        if not any(k in lower_name for k in ('point', 'pixel', 'coord', 'coordinate', '@', ' at ')):
            return None

        point_match = re.search(r'(-?\d+)\s*[,，]\s*(-?\d+)', action_name)
        if point_match is None:
            return None
        x = int(point_match.group(1))
        y = int(point_match.group(2))
        if 'turn on' in lower_name or 'toggle on' in lower_name:
            action_type = "toggleon_by_point"
        elif 'turn off' in lower_name or 'toggle off' in lower_name:
            action_type = "toggleoff_by_point"
        elif 'put down' in lower_name or 'put' in lower_name or 'place' in lower_name:
            action_type = "put_by_point"
        elif 'open' in lower_name:
            action_type = "open_by_point"
        elif 'close' in lower_name:
            action_type = "close_by_point"
        elif 'slice' in lower_name:
            action_type = "slice_by_point"
        elif 'pick' in lower_name:
            action_type = "pickup_by_point"
        else:
            return None
        return {"action": action_type, "point": [x, y]}

    @staticmethod
    def _extract_answer_block(output_text):
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, flags=re.IGNORECASE | re.DOTALL)
        if answer_match is not None:
            return answer_match.group(1).strip()
        return output_text.strip()

    def _parse_easyr1_json_action(self, answer_text):
        answer_text = self._strip_code_fence(answer_text)
        try:
            parsed = json.loads(answer_text)
        except json.JSONDecodeError:
            return None

        payload = parsed
        if isinstance(parsed, list):
            if len(parsed) == 0:
                return -1
            payload = parsed[0]
        if not isinstance(payload, dict):
            return None

        action_type = str(payload.get('action_type', payload.get('action', ''))).strip()
        parameter = payload.get('parameter', payload.get('action_param', payload.get('point')))
        if not action_type:
            return None

        action_key = action_type.lower()
        if action_key in ('move', 'rotate', 'look'):
            if not isinstance(parameter, str):
                return -1
            nav_action = self._EASYR1_JSON_NAV_MAP.get((action_key, parameter.strip().lower()))
            return nav_action if nav_action is not None else -1

        if action_key in self._EASYR1_JSON_INTER_MAP:
            point = self._normalize_easyr1_point(parameter)
            if point is None:
                return -1
            return {"action": self._EASYR1_JSON_INTER_MAP[action_key], "point": point}

        if action_type in self._EASYR1_NAV_ACTIONS:
            nav_type, nav_param = self._EASYR1_NAV_ACTIONS[action_type]
            return self._EASYR1_JSON_NAV_MAP.get((nav_type.lower(), nav_param.lower()), -1)

        return None

    def _parse_easyr1_output(self, output_text):
        answer_text = self._extract_answer_block(output_text)
        answer_text = answer_text.replace('\n', ' ').strip()

        parsed_json_action = self._parse_easyr1_json_action(answer_text)
        if parsed_json_action is not None:
            return parsed_json_action

        nav_map = {
            'moveahead_25': "MoveAhead",
            'moveahead': "MoveAhead",
            'moveback_25': "MoveBack",
            'moveback': "MoveBack",
            'rotateright_90': "RotateRight",
            'rotateright': "RotateRight",
            'rotateleft_90': "RotateLeft",
            'rotateleft': "RotateLeft",
            'lookdown_15': "LookDown_15",
            'lookdown': "LookDown_15",
            'lookup_15': "LookUp_15",
            'lookup': "LookUp_15",
        }
        inter_map = {
            'pickupobject': 'pickup_by_point',
            'putobject': 'put_by_point',
            'closeobject': 'close_by_point',
            'openobject': 'open_by_point',
            'toggleobjecton': 'toggleon_by_point',
            'toggleobjectoff': 'toggleoff_by_point',
            'sliceobject': 'slice_by_point',
        }

        # Action token is usually the first token in <answer>.
        token_match = re.search(
            r'(MoveAhead_25|MoveAhead|MoveBack_25|MoveBack|RotateRight_90|RotateRight|RotateLeft_90|RotateLeft|LookDown_15|LookDown|LookUp_15|LookUp|PickupObject|PutObject|CloseObject|OpenObject|ToggleObjectOn|ToggleObjectOff|SliceObject)',
            answer_text,
            flags=re.IGNORECASE
        )
        if token_match is None:
            return None

        action_token = token_match.group(1).strip()
        action_key = action_token.lower()
        if action_key in nav_map:
            return nav_map[action_key]

        if action_key in inter_map:
            point_match = re.search(r'[\[\(]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[\]\)]', answer_text)
            if point_match is None:
                return -1
            x = int(point_match.group(1))
            y = int(point_match.group(2))
            return {"action": inter_map[action_key], "point": [x, y]}

        return None
    

    def _materialize_image_path(self, image, prefix):
        if image is None:
            return None
        if isinstance(image, str):
            return image

        os.makedirs('./evaluation', exist_ok=True)
        image_path = './evaluation/tmp_{}_{}_{}.png'.format(prefix, self.planner_steps, len(prefix))
        cv2.imwrite(image_path, image)
        return image_path

    def _get_topdown_image_paths(self, image):
        if not isinstance(image, dict):
            raise ValueError("Topdown prompt expects an observation dict with head_rgb and topdown_rgb.")

        head_path = self._materialize_image_path(image.get(self.obs_key), self.obs_key)
        topdown_path = self._materialize_image_path(image.get(self.topdown_obs_key), self.topdown_obs_key)
        if head_path is None or topdown_path is None:
            raise ValueError("Missing head_rgb or topdown_rgb for topdown prompt.")
        return [head_path, topdown_path]

    def _get_custom_model_observation(self, obs):
        if self.use_topdown_prompt:
            return self._get_topdown_image_paths(obs)

        if isinstance(obs, dict):
            obs = obs.get(self.obs_key)
        return self._materialize_image_path(obs, self.obs_key)

    @staticmethod
    def _build_interleaved_content(prompt, image_data_urls):
        """Split prompt on <image> placeholders and interleave image_url objects."""
        segments = prompt.split('<image>')
        content = []
        for i, seg in enumerate(segments):
            if seg.strip():
                content.append({"type": "text", "text": seg})
            if i < len(image_data_urls):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data_urls[i]}
                })
        return content

    @staticmethod
    def _build_interleaved_raw_content(prompt, raw_images):
        """Split prompt on <image> placeholders and interleave raw image objects."""
        segments = prompt.split('<image>')
        content = []
        for i, seg in enumerate(segments):
            if seg.strip():
                text = seg.rstrip("\n") if i == len(segments) - 1 else seg
                content.append({"type": "text", "text": text})
            if i < len(raw_images):
                content.append({
                    "type": "image",
                    "image": raw_images[i],
                })
        return content

    def _materialize_direct_image(self, image):
        if image is None:
            return None
        if isinstance(image, str):
            from PIL import Image

            return Image.open(image).convert("RGB")
        return image

    def _get_topdown_raw_images(self, image):
        if not isinstance(image, dict):
            raise ValueError("Topdown prompt expects an observation dict with head_rgb and topdown_rgb.")

        head_image = self._materialize_direct_image(image.get(self.obs_key))
        topdown_image = self._materialize_direct_image(image.get(self.topdown_obs_key))
        if head_image is None or topdown_image is None:
            raise ValueError("Missing head_rgb or topdown_rgb for topdown prompt.")
        return [head_image, topdown_image]

    def get_message(self, image, prompt, messages=[]):
        if self.language_only:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}],
                }
            ]
        else:
            if self.model_type == 'vllm_direct':
                if self.use_topdown_prompt:
                    raw_images = self._get_topdown_raw_images(image)
                    if self.use_easyr1_format and '<image>' in prompt:
                        content = self._build_interleaved_raw_content(prompt, raw_images)
                    else:
                        content = [{"type": "image", "image": img} for img in raw_images]
                        content.append({"type": "text", "text": prompt})
                    return messages + [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]

                if isinstance(image, dict):
                    image = image.get(self.obs_key)

                if self.multistep and isinstance(image, (list, tuple)):
                    raw_images = [self._materialize_direct_image(img) for img in image[-self.multistep:]]
                else:
                    raw_image = self._materialize_direct_image(image)
                    raw_images = [raw_image] if raw_image is not None else []

                if self.use_easyr1_format and '<image>' in prompt:
                    content = self._build_interleaved_raw_content(prompt, raw_images)
                else:
                    content = [{"type": "image", "image": img} for img in raw_images]
                    content.append({"type": "text", "text": prompt})

                return messages + [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]

            if self.use_topdown_prompt:
                image_data_urls = [
                    local_image_to_data_url(image_path=p)
                    for p in self._get_topdown_image_paths(image)
                ]
                if self.use_easyr1_format and '<image>' in prompt:
                    content = self._build_interleaved_content(prompt, image_data_urls)
                else:
                    content = []
                    for url in image_data_urls:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
                    content.append({"type": "text", "text": prompt})
                return messages + [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]

            if type(image) == str:
                image_path = image 
            elif self.multistep and isinstance(image, (list, tuple)):
                image_path = image[-1] if len(image) else ''
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            if self.multistep: # handle multiple images
                content = [{"type": "text", "text": prompt}]
                if isinstance(image, (list, tuple)):
                    image_paths = [img for img in image if isinstance(img, str)]
                    for temp_path in image_paths[-self.multistep:]:
                        temp_data_url = local_image_to_data_url(image_path=temp_path)
                        content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": temp_data_url,
                                }})
                else:
                    ind = int(image_path.split('step_')[-1].strip('.png'))
                    for i in range(max(ind - self.multistep + 1, 0), ind +1):
                        temp_path = ''.join(image_path.split('step_')[:-1])+ f'step_{str(i)}.png'
                        temp_data_url = local_image_to_data_url(image_path=temp_path)
                        content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": temp_data_url,
                                }})
            else:
                data_url = local_image_to_data_url(image_path=image_path)
                if self.use_easyr1_format and '<image>' in prompt:
                    content = self._build_interleaved_content(prompt, [data_url])
                else:
                    content = [{ "type": "image_url", "image_url": { "url": data_url,}}, {"type": "text", "text": prompt}]

            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.episode_memory = ""
        self.episode_inventory = ""
        self.steps_since_memory_refresh = 0
        self.planner_steps = 0
        self.output_json_error = 0
        # EasyR1 v4-summary recursion state. Cleared every episode so prompts
        # for step 0 fall back to the "(none, first step)" / "none (first step)"
        # defaults declared in alfred.jinja.
        self.prev_summary = ""
        self.prev_action_desc = ""
        self.frames_unchanged_hint = ""

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action
    
    def json_to_action(self, output_text, json_key='executable_plan'):
        try:
            json_object = json.loads(output_text)
            action = []
            for i, step in enumerate(json_object[json_key]):
                point_action = self._parse_point_action(step) if self.enable_point_actions else None
                if self.enable_point_actions and point_action is not None:
                    action.append(point_action)
                    continue

                if self.action_key not in step:
                    print(f'missing action key in step {i}')
                    if i == 0:
                        action = -1
                    break

                act_id = step[self.action_key]
                if not isinstance(act_id, int):
                    print(f'invalid action id type in step {i}')
                    if i == 0:
                        action = -1
                    break
                action.append(act_id)

            if not len(action):
                print('empty plan, stop here')
                action = -2
            else:
                # keep action valid
                for i, act in enumerate(action):
                    if not isinstance(act, int):
                        continue
                    if act >= len(self.actions) or act < 0:
                        print('found invlid action')
                        if i == 0:
                            action = -1
                        else:
                            action = action[:i]
                        break
        except json.JSONDecodeError as e:
            if self.use_easyr1_format:
                parsed = self._parse_easyr1_output(output_text)
                if parsed is not None:
                    return parsed
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            if self.use_easyr1_format:
                parsed = self._parse_easyr1_output(output_text)
                if parsed is not None:
                    return parsed
            print("An unexpected error occurred:", e)
            traceback.print_exc()
            self.output_json_error += 1
            action = -1
        return action

    
        
    def _update_summary_state(self, output_text):
        """Capture the model's <summary> block and last-action description.

        Skipped for the navigation-only easyr1 variant whose template lacks a
        summary section. For the alfred summary template these fields feed
        back into the next step's prompt so the model recurses on its own
        prior progress/spatial summary, matching the EasyR1 v4 setup.
        """
        if not self.use_easyr1_format or self.easyr1_navigation_only:
            return
        new_summary = self._extract_summary_block(output_text)
        if new_summary:
            self.prev_summary = new_summary
        new_action_desc = self._last_action_from_output(output_text)
        if new_action_desc:
            self.prev_action_desc = new_action_desc

    def act_custom(self, prompt, obs):
        out = self.model.respond(prompt, self._get_custom_model_observation(obs))
        print(f"Model Output:\n{out}\n")
        # fix common generated json errors
        if not self.use_easyr1_format:
            out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action = self.json_to_action(out)
        self._update_summary_state(out)
        self.planner_steps += 1
        return action, out


    def act(self, observation, user_instruction):
        if type(observation) == dict:
            obs = observation if self.use_topdown_prompt else observation[self.obs_key]
        else:
            obs = observation # input image path
        
        prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        # some models do not support json scheme, add style into prompt
        if (not self.use_easyr1_format) and ('claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or 'Qwen2.5-VL' in self.model_name or self.model_type == 'custom'):
            prompt = prompt + template_lang if self.language_only else prompt + template
        print(f"Full Prompt:\n{prompt}\n")

        if self.model_type == 'custom':
            return self.act_custom(prompt, obs) 

        if self.memory_compression:
            self.episode_messages = self.get_message(obs, prompt)
        elif len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try: 
                out = self.model.respond(self.episode_messages)
                time.sleep(15)
            except Exception as e:
                print("An unexpected error occurred:", e)
                traceback.print_exc()
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            try: 
                out = self.model.respond(self.episode_messages)
            except Exception as e:
                print("An unexpected error occurred:", e)
                traceback.print_exc()

                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        print(f"Model Output:\n{out}\n")
        logger.debug(f"Model Output:\n{out}\n")

        if self.chat_history and not self.memory_compression:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
        action = self.json_to_action(out)
        self._update_summary_state(out)
        self.planner_steps += 1
        return action, out

    def update_info(self, info, previous_obs=None, current_obs=None):
        """Update episode feedback history."""
        # Update the stuck-frame hint regardless of mode; it's only consumed by
        # the alfred summary template, but tracking it here keeps reset/update
        # paths consistent and makes the behaviour testable in isolation.
        if previous_obs is not None and current_obs is not None and \
                self._observations_are_unchanged(previous_obs, current_obs):
            self.frames_unchanged_hint = self._stuck_hint_text()
        else:
            self.frames_unchanged_hint = ""

        if self.memory_compression:
            self.steps_since_memory_refresh += 1
            if self.steps_since_memory_refresh >= self.segment_len:
                self._rebuild_episode_memory(previous_obs, current_obs, info)
                self.steps_since_memory_refresh = 0
            return

        action_ref = info.get('action_description', info.get('action_id'))
        self.episode_act_feedback.append([
            action_ref,
            info['env_feedback'],
            info.get('last_action_success')
        ])


        

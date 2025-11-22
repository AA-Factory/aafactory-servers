from dataclasses import dataclass
from wan_animate.wan_animate.cli_arguments.utils import to_int, to_float, to_str, to_bool, to_list

ARGUMENTS_REPLACE = {
    # ------ User Arguments ------
    "person_index": {
        "map_to": "value52",
        "type": to_int,
        "help": "Which person to replace (0-indexed).",
    },
    "blocks_to_swap": {
        "map_to": "blocks_to_swap45",
        "type": to_int,
        "help": "Number of blocks to swap to RAM (max 40)",
    },
    "prefetch_blocks": {
        "map_to": "prefetch_blocks50",
        "type": to_int,
        "help": "Number of blocks to prefetch. (1 if blocks_to_swap<40)",
    },
    "block_swap_debug": {
        "map_to": "block_swap_debug51",
        "type": to_bool,
        "help": "Enable block swap debug mode",
    },
    "steps": {
        "map_to": "steps132",
        "type": to_int,
        "help": "Number of sampling steps.",
    },
    "cfg": {
        "map_to": "cfg133",
        "type": to_int,
        "help": "Classifier-Free Guidance scale.",
    },
    "shift": {
        "map_to": "shift134",
        "type": to_int,
        "help": "Shift value.",
    },
    "divide_fws": {
        "map_to": "value36",
        "type": to_int,
        "help": "Divide FWS value.",
    },
    "interpolation_factor": {
        "map_to": "value1",
        "type": to_int,
        "help": "Interpolation factor.",
    },
    "framerate": {
        "map_to": "value2",
        "type": to_int,
        "help": "Output framerate (FPS). Will be interpolated.",
    },
    "resolution": {
        "map_to": "value3",
        "type": to_int,
        "help": "Output resolution, e.g., '720'.",
    },
    "positive_prompt": {
        "map_to": "text35",
        "type": to_str,
        "help": "The positive text prompt.",
    },
    "negative_prompt": {
        "map_to": "text34",
        "type": to_str,
        "help": "The negative text prompt.",
    },
    # ------ System Arguments ------
    "input_image": {
        "map_to": "image8",
        "type": to_str,
        "help": "Input image file name.",
    },
    "input_video": {
        "map_to": ["video37", "video54"],
        "type": to_str,
        "help": "Input video file name.",
    },
}

SYSTEM_ONLY_KEYS = {"input_image", "input_video"}


@dataclass
class ArgumentConfigReplace:
    arguments = ARGUMENTS_REPLACE
    system_keys = SYSTEM_ONLY_KEYS
    user_keys = set(arguments.keys()) - system_keys

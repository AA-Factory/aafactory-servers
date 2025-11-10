from wan_animate.arg_utils import to_int, to_float, to_str, to_bool, to_list

ARG_CONFIG = {
    # ------ User Arguments ------
    "blocks_to_swap": {
        "map_to": "blocks_to_swap41",
        "type": to_int,
        "help": "Number of blocks to swap to RAM",
    },
    "prefetch_blocks": {
        "map_to": "prefetch_blocks46",
        "type": to_int,
        "help": "Number of blocks to prefetch.",
    },
    "block_swap_debug": {
        "map_to": "block_swap_debug47",
        "type": to_bool,
        "help": "Enable block swap debug mode",
    },
    "steps": {
        "map_to": "steps110",
        "type": to_int,
        "help": "Number of sampling steps.",
    },
    "cfg": {
        "map_to": "cfg111",
        "type": to_int,
        "help": "Classifier-Free Guidance scale.",
    },
    "shift": {
        "map_to": "shift112",
        "type": to_int,
        "help": "Shift value.",
    },
    "divide_fws": {
        "map_to": "value32",
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
        "help": "Output framerate (FPS).",
    },
    "resolution": {
        "map_to": "value3",
        "type": to_int,
        "help": "Output resolution, e.g., '720'.",
    },
    "positive_prompt": {
        "map_to": "text31",
        "type": to_str,
        "help": "The positive text prompt.",
    },
    "negative_prompt": {
        "map_to": "text30",
        "type": to_str,
        "help": "The negative text prompt.",
    },
    # ------ System Arguments ------
    "input_image": {
        "map_to": "image4",
        "type": to_str,
        "help": "Input image file name.",
    },
    "input_video": {
        "map_to": "video33",
        "type": to_str,
        "help": "Input video file name.",
    },
}

SYSTEM_ONLY_KEYS = {"input_image", "input_video"}

USER_SETTABLE_KEYS = set(ARG_CONFIG.keys()) - SYSTEM_ONLY_KEYS

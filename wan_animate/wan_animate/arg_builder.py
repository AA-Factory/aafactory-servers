import os
import logging
from typing import Any, Dict, List

from wan_animate.arg_config import ARG_CONFIG, SYSTEM_ONLY_KEYS, USER_SETTABLE_KEYS

def _validate_and_map_args(
    args: Dict[str, Any], allowed_keys: set[str], strict: bool = True
) -> Dict[str, Any]:
    """
    Validates, coerces, and maps args to their target names, but only for a
    specific set of allowed keys.
    """
    mapped_args: Dict[str, Any] = {}
    for user_key, user_value in args.items():
        if user_key not in allowed_keys:
            if strict:
                raise ValueError(f"Unknown or forbidden argument: {user_key}")
            else:
                continue

        config = ARG_CONFIG[user_key]
        mapped_key = config["map_to"]
        coerce_func = config["type"]

        try:
            coerced_value = coerce_func(user_value)
            mapped_args[mapped_key] = coerced_value
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Invalid value for argument '{user_key}': {user_value}. {e}"
            ) from e

    return mapped_args


def _dict_to_cli_args(overrides: Dict[str, Any]) -> List[str]:
    """
    Convert override dict to CLI args. Only keys present are emitted.
    Boolean: emits --key (True) or --no-key (False)
    List: emits --key val1 --key val2  (or change to single csv if your generator expects that)
    Others: emit --key value
    """
    out: List[str] = []
    for k, v in overrides.items():
        flag = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            out.append(flag if v else f"--no-{k.replace('_','-')}")
        elif isinstance(v, (list, tuple)):
            for item in v:
                out.extend([flag, str(item)])
        else:
            out.extend([flag, str(v)])
    return out

def build_user_cli_args(user_args: Dict[str, Any]) -> List[str]:
    """
    Converts a user-provided dict into CLI arguments, ensuring only
    user-settable arguments are processed.
    """
    try:
        # Pass the specific set of keys allowed for users
        mapped_args = _validate_and_map_args(
            user_args, allowed_keys=USER_SETTABLE_KEYS, strict=True
        )
    except (ValueError, TypeError) as e:
        logging.getLogger().error(f"Invalid user arguments: {e}")
        raise
    return _dict_to_cli_args(mapped_args)


def build_system_cli_args(image_path: str, video_path: str) -> List[str]:
    """
    Builds system-level arguments from the protected set of system keys.
    """
    system_args_dict = {
        "input_image": os.path.basename(image_path),
        "input_video": os.path.basename(video_path),
    }
    # Pass the specific set of keys reserved for the system
    mapped_args = _validate_and_map_args(
        system_args_dict, allowed_keys=SYSTEM_ONLY_KEYS, strict=True
    )
    return _dict_to_cli_args(mapped_args)
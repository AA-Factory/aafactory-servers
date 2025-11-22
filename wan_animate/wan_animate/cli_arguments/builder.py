import json
import os
import logging
from typing import Any, Dict, List

from wan_animate.wan_animate.cli_arguments.config_animate import ArgumentConfigAnimate
from wan_animate.wan_animate.cli_arguments.config_replace import ArgumentConfigReplace

class ArgumentBuilder:
    def __init__(self, config: ArgumentConfigAnimate | ArgumentConfigReplace):
        self.arguments = config.arguments
        self.system_keys = config.system_keys
        self.user_keys = config.user_keys


    def _validate_and_map_args(
        self, args: Dict[str, Any], allowed_keys: set[str], strict: bool = True
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

            argument = self.arguments[user_key]
            map_arg_to = argument["map_to"]
            coerce_func = argument["type"]

            try:
                coerced_value = coerce_func(user_value)
                if isinstance(map_arg_to, list):
                    for mapping_key in map_arg_to:
                        mapped_args[mapping_key] = coerced_value
                else:
                    mapped_args[map_arg_to] = coerced_value
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Invalid value for argument '{user_key}': {user_value}. {e}"
                ) from e

        return mapped_args


    def _dict_to_cli_args(self, overrides: Dict[str, Any]) -> List[str]:
        """
        Convert override dict to CLI args. Only keys present are emitted.
        Boolean: emits --key (True) or --no-key (False)
        List: emits --key val1 --key val2  (or change to single csv if your generator expects that)
        Others: emit --key value
        """
        out: List[str] = []
        for k, v in overrides.items():
            out.extend([f"--{k}", json.dumps(v)])
        return out


    def build_user_cli_args(self, user_args: Dict[str, Any]) -> List[str]:
        """
        Converts a user-provided dict into CLI arguments, ensuring only
        user-settable arguments are processed.
        """
        try:
            # Pass the specific set of keys allowed for users
            mapped_args = self._validate_and_map_args(
                user_args, allowed_keys=self.user_keys, strict=True
            )
        except (ValueError, TypeError) as e:
            logging.getLogger().error(f"Invalid user arguments: {e}")
            raise
        return self._dict_to_cli_args(mapped_args)


    def build_system_cli_args(self, image_path: str, video_path: str) -> List[str]:
        """
        Builds system-level arguments from the protected set of system keys.
        """
        system_args_dict = {
            "input_image": os.path.basename(image_path),
            "input_video": os.path.basename(video_path),
        }
        # Pass the specific set of keys reserved for the system
        mapped_args = self._validate_and_map_args(
            system_args_dict, allowed_keys=self.system_keys, strict=True
        )
        return self._dict_to_cli_args(mapped_args)

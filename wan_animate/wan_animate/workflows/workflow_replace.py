import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path or not parent_directory:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)

        manager_path = os.path.join(
            comfyui_path, "custom_nodes", "ComfyUI-Manager", "glob"
        )

        global has_manager
        if os.path.isdir(manager_path) and os.listdir(manager_path):
            sys.path.append(manager_path)
            has_manager = True
        else:
            has_manager = False

        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from comfy.options import enable_args_parsing

    enable_args_parsing()
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    if has_manager:
        try:
            import manager_core as manager
        except ImportError:
            print("Could not import manager_core, proceeding without it.")
            return
        else:
            if hasattr(manager, "get_config"):
                print("Patching manager_core.get_config to enforce offline mode.")
                try:
                    get_config = manager.get_config

                    def _get_config(*args, **kwargs):
                        config = get_config(*args, **kwargs)
                        config["network_mode"] = "offline"
                        return config

                    manager.get_config = _get_config
                except Exception as e:
                    print("Failed to patch manager_core.get_config:", e)

    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def inner():
        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        await init_extra_nodes(init_custom_nodes=True)

    loop.run_until_complete(inner())


def save_image_wrapper(context, cls):
    if args.output is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output == "-":
                    raise ValueError("Cannot save multiple images to stdout")
                filename_prefix += self.prefix_append

                results = list()
                for batch_number, image in enumerate(images):
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    if args.output == "-":
                        # Hack to briefly restore stdout
                        if context is not None:
                            context.__exit__(None, None, None)
                        try:
                            img.save(
                                sys.stdout.buffer,
                                format="png",
                                pnginfo=metadata,
                                compress_level=self.compress_level,
                            )
                        finally:
                            if context is not None:
                                context.__enter__()
                    else:
                        subfolder = ""
                        if len(images) == 1:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output)

                            if subfolder == "":
                                subfolder = os.getcwd()

                            files = os.listdir(subfolder)
                            file_pattern = file
                            while True:
                                filename_with_batch_num = file_pattern.replace(
                                    "%batch_num%", str(batch_number)
                                )
                                file = (
                                    f"{filename_with_batch_num}_{self.counter:05}.png"
                                )
                                self.counter += 1

                                if file not in files:
                                    break

                        img.save(
                            os.path.join(subfolder, file),
                            pnginfo=metadata,
                            compress_level=self.compress_level,
                        )
                        print("Saved image to", os.path.join(subfolder, file))
                        results.append(
                            {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type,
                            }
                        )

                return {"ui": {"images": results}}

    return WrappedSaveImage


def parse_arg(s: Any, default: Any = None) -> Any:
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Node inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "--value1",
    default=2,
    help='Argument 0, input `value` for node "🔀 Interpolation factor" id 29 (autogenerated)',
)

parser.add_argument(
    "--value2",
    default=16,
    help='Argument 0, input `value` for node "🎦 Framerate" id 30 (autogenerated)',
)

parser.add_argument(
    "--value3",
    default=720,
    help='Argument 0, input `value` for node "🔛 Resolution" id 32 (autogenerated)',
)

parser.add_argument(
    "--model4",
    default="sam2_hiera_base_plus.safetensors",
    help='Argument 0, input `model` for node "(Down)Load SAM2Model" id 36 (autogenerated)',
)

parser.add_argument(
    "--segmentor5",
    default="video",
    help='Argument 1, input `segmentor` for node "(Down)Load SAM2Model" id 36 (autogenerated)',
)

parser.add_argument(
    "--device6",
    default="cuda",
    help='Argument 2, input `device` for node "(Down)Load SAM2Model" id 36 (autogenerated)',
)

parser.add_argument(
    "--precision7",
    default="fp16",
    help='Argument 3, input `precision` for node "(Down)Load SAM2Model" id 36 (autogenerated)',
)

parser.add_argument(
    "--image8",
    default="image.jpeg",
    help='Argument 0, input `image` for node "Load Image" id 64 (autogenerated)',
)

parser.add_argument(
    "--lora_09",
    default="WanAnimate_relight_lora_fp16.safetensors",
    help='Argument 0, input `lora_0` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--strength_010",
    default=1,
    help='Argument 1, input `strength_0` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--lora_111",
    default="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    help='Argument 2, input `lora_1` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--strength_112",
    default=1.2,
    help='Argument 3, input `strength_1` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--lora_213",
    default="none",
    help='Argument 4, input `lora_2` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--strength_214",
    default=1,
    help='Argument 5, input `strength_2` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--lora_315",
    default="none",
    help='Argument 6, input `lora_3` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--strength_316",
    default=1,
    help='Argument 7, input `strength_3` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--lora_417",
    default="none",
    help='Argument 8, input `lora_4` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--strength_418",
    default=1,
    help='Argument 9, input `strength_4` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--low_mem_load19",
    default=False,
    help='Argument 12, input `low_mem_load` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--merge_loras20",
    default=False,
    help='Argument 13, input `merge_loras` for node "WanVideo Lora Select Multi" id 76 (autogenerated)',
)

parser.add_argument(
    "--clip_name21",
    default="clip_vision_h.safetensors",
    help='Argument 0, input `clip_name` for node "Load CLIP Vision" id 80 (autogenerated)',
)

parser.add_argument(
    "--model_name22",
    default="Wan2_1_VAE_bf16.safetensors",
    help='Argument 0, input `model_name` for node "WanVideo VAE Loader" id 81 (autogenerated)',
)

parser.add_argument(
    "--precision23",
    default="fp16",
    help='Argument 1, input `precision` for node "WanVideo VAE Loader" id 81 (autogenerated)',
)

parser.add_argument(
    "--backend24",
    default="inductor",
    help='Argument 0, input `backend` for node "WanVideo Torch Compile Settings" id 82 (autogenerated)',
)

parser.add_argument(
    "--fullgraph25",
    default=False,
    help='Argument 1, input `fullgraph` for node "WanVideo Torch Compile Settings" id 82 (autogenerated)',
)

parser.add_argument(
    "--mode26",
    default="default",
    help='Argument 2, input `mode` for node "WanVideo Torch Compile Settings" id 82 (autogenerated)',
)

parser.add_argument(
    "--dynamic27",
    default=False,
    help='Argument 3, input `dynamic` for node "WanVideo Torch Compile Settings" id 82 (autogenerated)',
)

parser.add_argument(
    "--dynamo_cache_size_limit28",
    default=64,
    help='Argument 4, input `dynamo_cache_size_limit` for node "WanVideo Torch Compile Settings" id 82 (autogenerated)',
)

parser.add_argument(
    "--compile_transformer_blocks_only29",
    default=True,
    help='Argument 5, input `compile_transformer_blocks_only` for node "WanVideo Torch Compile Settings" id 82 (autogenerated)',
)

parser.add_argument(
    "--dynamo_recompile_limit30",
    default=128,
    help='Argument 6, input `dynamo_recompile_limit` for node "WanVideo Torch Compile Settings" id 82 (autogenerated)',
)

parser.add_argument(
    "--clip_name31",
    default="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    help='Argument 0, input `clip_name` for node "Load CLIP" id 119 (autogenerated)',
)

parser.add_argument(
    "--type32",
    default="wan",
    help='Argument 1, input `type` for node "Load CLIP" id 119 (autogenerated)',
)

parser.add_argument(
    "--device33",
    default="default",
    help='Argument 2, input `device` for node "Load CLIP" id 119 (autogenerated)',
)

parser.add_argument(
    "--text34",
    default="过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, tattoo,",
    help='Argument 0, input `text` for node "CLIP Text Encode (Prompt)" id 122 (autogenerated)',
)

parser.add_argument(
    "--text35",
    default="",
    help='Argument 0, input `text` for node "CLIP Text Encode (Prompt)" id 124 (autogenerated)',
)

parser.add_argument(
    "--value36",
    default=1,
    help='Argument 0, input `value` for node "🔣 Divide FWS" id 161 (autogenerated)',
)

parser.add_argument(
    "--video37",
    default="video.mp4",
    help='Argument 0, input `video` for node "Load Video (Upload) 🎥🅥🅗🅢" id 222 (autogenerated)',
)

parser.add_argument(
    "--force_rate38",
    default=0,
    help='Argument 1, input `force_rate` for node "Load Video (Upload) 🎥🅥🅗🅢" id 222 (autogenerated)',
)

parser.add_argument(
    "--custom_width39",
    default=0,
    help='Argument 2, input `custom_width` for node "Load Video (Upload) 🎥🅥🅗🅢" id 222 (autogenerated)',
)

parser.add_argument(
    "--custom_height40",
    default=0,
    help='Argument 3, input `custom_height` for node "Load Video (Upload) 🎥🅥🅗🅢" id 222 (autogenerated)',
)

parser.add_argument(
    "--frame_load_cap41",
    default=65,
    help='Argument 4, input `frame_load_cap` for node "Load Video (Upload) 🎥🅥🅗🅢" id 222 (autogenerated)',
)

parser.add_argument(
    "--skip_first_frames42",
    default=0,
    help='Argument 5, input `skip_first_frames` for node "Load Video (Upload) 🎥🅥🅗🅢" id 222 (autogenerated)',
)

parser.add_argument(
    "--select_every_nth43",
    default=1,
    help='Argument 6, input `select_every_nth` for node "Load Video (Upload) 🎥🅥🅗🅢" id 222 (autogenerated)',
)

parser.add_argument(
    "--format44",
    default="Wan",
    help='Argument 9, input `format` for node "Load Video (Upload) 🎥🅥🅗🅢" id 222 (autogenerated)',
)

parser.add_argument(
    "--blocks_to_swap45",
    default=40,
    help='Argument 0, input `blocks_to_swap` for node "WanVideo Block Swap" id 238 (autogenerated)',
)

parser.add_argument(
    "--offload_img_emb46",
    default=False,
    help='Argument 1, input `offload_img_emb` for node "WanVideo Block Swap" id 238 (autogenerated)',
)

parser.add_argument(
    "--offload_txt_emb47",
    default=False,
    help='Argument 2, input `offload_txt_emb` for node "WanVideo Block Swap" id 238 (autogenerated)',
)

parser.add_argument(
    "--use_non_blocking48",
    default=False,
    help='Argument 3, input `use_non_blocking` for node "WanVideo Block Swap" id 238 (autogenerated)',
)

parser.add_argument(
    "--vace_blocks_to_swap49",
    default=0,
    help='Argument 4, input `vace_blocks_to_swap` for node "WanVideo Block Swap" id 238 (autogenerated)',
)

parser.add_argument(
    "--prefetch_blocks50",
    default=0,
    help='Argument 5, input `prefetch_blocks` for node "WanVideo Block Swap" id 238 (autogenerated)',
)

parser.add_argument(
    "--block_swap_debug51",
    default=False,
    help='Argument 6, input `block_swap_debug` for node "WanVideo Block Swap" id 238 (autogenerated)',
)

parser.add_argument(
    "--value52",
    default=0,
    help='Argument 0, input `value` for node "Person Index" id 263 (autogenerated)',
)

parser.add_argument(
    "--expression53",
    default="(a*b)+1",
    help='Argument 0, input `expression` for node "Calculate Frames" id 21 (autogenerated)',
)

parser.add_argument(
    "--video54",
    default="video.mp4",
    help='Argument 0, input `video` for node "Load Video (Upload) 🎥🅥🅗🅢" id 13 (autogenerated)',
)

parser.add_argument(
    "--custom_width55",
    default=0,
    help='Argument 2, input `custom_width` for node "Load Video (Upload) 🎥🅥🅗🅢" id 13 (autogenerated)',
)

parser.add_argument(
    "--custom_height56",
    default=0,
    help='Argument 3, input `custom_height` for node "Load Video (Upload) 🎥🅥🅗🅢" id 13 (autogenerated)',
)

parser.add_argument(
    "--skip_first_frames57",
    default=0,
    help='Argument 5, input `skip_first_frames` for node "Load Video (Upload) 🎥🅥🅗🅢" id 13 (autogenerated)',
)

parser.add_argument(
    "--select_every_nth58",
    default=1,
    help='Argument 6, input `select_every_nth` for node "Load Video (Upload) 🎥🅥🅗🅢" id 13 (autogenerated)',
)

parser.add_argument(
    "--format59",
    default="Wan",
    help='Argument 9, input `format` for node "Load Video (Upload) 🎥🅥🅗🅢" id 13 (autogenerated)',
)

parser.add_argument(
    "--height60",
    default=10000,
    help='Argument 2, input `height` for node "Resize Image v2" id 33 (autogenerated)',
)

parser.add_argument(
    "--upscale_method61",
    default="lanczos",
    help='Argument 3, input `upscale_method` for node "Resize Image v2" id 33 (autogenerated)',
)

parser.add_argument(
    "--keep_proportion62",
    default="resize",
    help='Argument 4, input `keep_proportion` for node "Resize Image v2" id 33 (autogenerated)',
)

parser.add_argument(
    "--pad_color63",
    default="0, 0, 0",
    help='Argument 5, input `pad_color` for node "Resize Image v2" id 33 (autogenerated)',
)

parser.add_argument(
    "--crop_position64",
    default="center",
    help='Argument 6, input `crop_position` for node "Resize Image v2" id 33 (autogenerated)',
)

parser.add_argument(
    "--divisible_by65",
    default=16,
    help='Argument 7, input `divisible_by` for node "Resize Image v2" id 33 (autogenerated)',
)

parser.add_argument(
    "--device66",
    default="cpu",
    help='Argument 9, input `device` for node "Resize Image v2" id 33 (autogenerated)',
)

parser.add_argument(
    "--per_batch67",
    default=0,
    help='Argument 10, input `per_batch` for node "Resize Image v2" id 33 (autogenerated)',
)

parser.add_argument(
    "--frame_count68",
    default=1,
    help='Argument 1, input `frame_count` for node "Generate Frames By Count" id 9 (autogenerated)',
)

parser.add_argument(
    "--revert69",
    default=True,
    help='Argument 2, input `revert` for node "Generate Frames By Count" id 9 (autogenerated)',
)

parser.add_argument(
    "--height70",
    default=10000,
    help='Argument 2, input `height` for node "Resize Image v2" id 63 (autogenerated)',
)

parser.add_argument(
    "--upscale_method71",
    default="lanczos",
    help='Argument 3, input `upscale_method` for node "Resize Image v2" id 63 (autogenerated)',
)

parser.add_argument(
    "--keep_proportion72",
    default="resize",
    help='Argument 4, input `keep_proportion` for node "Resize Image v2" id 63 (autogenerated)',
)

parser.add_argument(
    "--pad_color73",
    default="0, 0, 0",
    help='Argument 5, input `pad_color` for node "Resize Image v2" id 63 (autogenerated)',
)

parser.add_argument(
    "--crop_position74",
    default="center",
    help='Argument 6, input `crop_position` for node "Resize Image v2" id 63 (autogenerated)',
)

parser.add_argument(
    "--divisible_by75",
    default=16,
    help='Argument 7, input `divisible_by` for node "Resize Image v2" id 63 (autogenerated)',
)

parser.add_argument(
    "--device76",
    default="cpu",
    help='Argument 9, input `device` for node "Resize Image v2" id 63 (autogenerated)',
)

parser.add_argument(
    "--per_batch77",
    default=0,
    help='Argument 10, input `per_batch` for node "Resize Image v2" id 63 (autogenerated)',
)

parser.add_argument(
    "--resize_mode78",
    default="Just Resize",
    help='Argument 3, input `resize_mode` for node "Pixel Perfect Resolution" id 58 (autogenerated)',
)

parser.add_argument(
    "--detect_hand79",
    default="enable",
    help='Argument 1, input `detect_hand` for node "DWPose Estimator" id 53 (autogenerated)',
)

parser.add_argument(
    "--detect_body80",
    default="enable",
    help='Argument 2, input `detect_body` for node "DWPose Estimator" id 53 (autogenerated)',
)

parser.add_argument(
    "--detect_face81",
    default="enable",
    help='Argument 3, input `detect_face` for node "DWPose Estimator" id 53 (autogenerated)',
)

parser.add_argument(
    "--bbox_detector82",
    default="yolox_l.torchscript.pt",
    help='Argument 5, input `bbox_detector` for node "DWPose Estimator" id 53 (autogenerated)',
)

parser.add_argument(
    "--pose_estimator83",
    default="dw-ll_ucoco_384_bs5.torchscript.pt",
    help='Argument 6, input `pose_estimator` for node "DWPose Estimator" id 53 (autogenerated)',
)

parser.add_argument(
    "--scale_stick_for_xinsr_cn84",
    default="disable",
    help='Argument 7, input `scale_stick_for_xinsr_cn` for node "DWPose Estimator" id 53 (autogenerated)',
)

parser.add_argument(
    "--top_k_pose85",
    default=5,
    help='Argument 2, input `top_k_pose` for node "PoseToSAMPoints" id 255 (autogenerated)',
)

parser.add_argument(
    "--top_k_face86",
    default=1,
    help='Argument 3, input `top_k_face` for node "PoseToSAMPoints" id 255 (autogenerated)',
)

parser.add_argument(
    "--top_k_left_hand87",
    default=1,
    help='Argument 4, input `top_k_left_hand` for node "PoseToSAMPoints" id 255 (autogenerated)',
)

parser.add_argument(
    "--top_k_right_hand88",
    default=1,
    help='Argument 5, input `top_k_right_hand` for node "PoseToSAMPoints" id 255 (autogenerated)',
)

parser.add_argument(
    "--keep_model_loaded89",
    default=False,
    help='Argument 2, input `keep_model_loaded` for node "Sam2Segmentation" id 37 (autogenerated)',
)

parser.add_argument(
    "--individual_objects90",
    default=False,
    help='Argument 6, input `individual_objects` for node "Sam2Segmentation" id 37 (autogenerated)',
)

parser.add_argument(
    "--expand91",
    default=10,
    help='Argument 1, input `expand` for node "GrowMask" id 38 (autogenerated)',
)

parser.add_argument(
    "--tapered_corners92",
    default=True,
    help='Argument 2, input `tapered_corners` for node "GrowMask" id 38 (autogenerated)',
)

parser.add_argument(
    "--block_size93",
    default=32,
    help='Argument 1, input `block_size` for node "Blockify Mask" id 39 (autogenerated)',
)

parser.add_argument(
    "--device94",
    default="cpu",
    help='Argument 2, input `device` for node "Blockify Mask" id 39 (autogenerated)',
)

parser.add_argument(
    "--color95",
    default="0, 0, 0",
    help='Argument 2, input `color` for node "Draw Mask On Image" id 43 (autogenerated)',
)

parser.add_argument(
    "--device96",
    default="cpu",
    help='Argument 3, input `device` for node "Draw Mask On Image" id 43 (autogenerated)',
)

parser.add_argument(
    "--frame_rate97",
    default=16,
    help='Argument 1, input `frame_rate` for node "Video Combine 🎥🅥🅗🅢" id 44 (autogenerated)',
)

parser.add_argument(
    "--loop_count98",
    default=0,
    help='Argument 2, input `loop_count` for node "Video Combine 🎥🅥🅗🅢" id 44 (autogenerated)',
)

parser.add_argument(
    "--filename_prefix99",
    default="WanVideo2_1_T2V",
    help='Argument 3, input `filename_prefix` for node "Video Combine 🎥🅥🅗🅢" id 44 (autogenerated)',
)

parser.add_argument(
    "--format100",
    default="video/h264-mp4",
    help='Argument 4, input `format` for node "Video Combine 🎥🅥🅗🅢" id 44 (autogenerated)',
)

parser.add_argument(
    "--pingpong101",
    default=False,
    help='Argument 5, input `pingpong` for node "Video Combine 🎥🅥🅗🅢" id 44 (autogenerated)',
)

parser.add_argument(
    "--save_output102",
    default=True,
    help='Argument 6, input `save_output` for node "Video Combine 🎥🅥🅗🅢" id 44 (autogenerated)',
)

parser.add_argument(
    "--frame_rate103",
    default=16,
    help='Argument 1, input `frame_rate` for node "Video Combine 🎥🅥🅗🅢" id 52 (autogenerated)',
)

parser.add_argument(
    "--loop_count104",
    default=0,
    help='Argument 2, input `loop_count` for node "Video Combine 🎥🅥🅗🅢" id 52 (autogenerated)',
)

parser.add_argument(
    "--filename_prefix105",
    default="WanVideo2_1_T2V",
    help='Argument 3, input `filename_prefix` for node "Video Combine 🎥🅥🅗🅢" id 52 (autogenerated)',
)

parser.add_argument(
    "--format106",
    default="video/h264-mp4",
    help='Argument 4, input `format` for node "Video Combine 🎥🅥🅗🅢" id 52 (autogenerated)',
)

parser.add_argument(
    "--pingpong107",
    default=False,
    help='Argument 5, input `pingpong` for node "Video Combine 🎥🅥🅗🅢" id 52 (autogenerated)',
)

parser.add_argument(
    "--save_output108",
    default=False,
    help='Argument 6, input `save_output` for node "Video Combine 🎥🅥🅗🅢" id 52 (autogenerated)',
)

parser.add_argument(
    "--base_resolution109",
    default=512,
    help='Argument 2, input `base_resolution` for node "Image Crop By Mask And Resize" id 56 (autogenerated)',
)

parser.add_argument(
    "--padding110",
    default=0,
    help='Argument 3, input `padding` for node "Image Crop By Mask And Resize" id 56 (autogenerated)',
)

parser.add_argument(
    "--min_crop_resolution111",
    default=128,
    help='Argument 4, input `min_crop_resolution` for node "Image Crop By Mask And Resize" id 56 (autogenerated)',
)

parser.add_argument(
    "--max_crop_resolution112",
    default=512,
    help='Argument 5, input `max_crop_resolution` for node "Image Crop By Mask And Resize" id 56 (autogenerated)',
)

parser.add_argument(
    "--model113",
    default="Wan2_2-Animate-14B_fp8_scaled_e5m2_KJ_v2.safetensors",
    help='Argument 0, input `model` for node "WanVideo Model Loader" id 75 (autogenerated)',
)

parser.add_argument(
    "--base_precision114",
    default="fp16_fast",
    help='Argument 1, input `base_precision` for node "WanVideo Model Loader" id 75 (autogenerated)',
)

parser.add_argument(
    "--quantization115",
    default="disabled",
    help='Argument 2, input `quantization` for node "WanVideo Model Loader" id 75 (autogenerated)',
)

parser.add_argument(
    "--load_device116",
    default="offload_device",
    help='Argument 3, input `load_device` for node "WanVideo Model Loader" id 75 (autogenerated)',
)

parser.add_argument(
    "--attention_mode117",
    default="sageattn",
    help='Argument 4, input `attention_mode` for node "WanVideo Model Loader" id 75 (autogenerated)',
)

parser.add_argument(
    "--rms_norm_function118",
    default="default",
    help='Argument 13, input `rms_norm_function` for node "WanVideo Model Loader" id 75 (autogenerated)',
)

parser.add_argument(
    "--strength_1119",
    default=1,
    help='Argument 2, input `strength_1` for node "WanVideo ClipVision Encode" id 87 (autogenerated)',
)

parser.add_argument(
    "--strength_2120",
    default=1,
    help='Argument 3, input `strength_2` for node "WanVideo ClipVision Encode" id 87 (autogenerated)',
)

parser.add_argument(
    "--crop121",
    default="center",
    help='Argument 4, input `crop` for node "WanVideo ClipVision Encode" id 87 (autogenerated)',
)

parser.add_argument(
    "--combine_embeds122",
    default="average",
    help='Argument 5, input `combine_embeds` for node "WanVideo ClipVision Encode" id 87 (autogenerated)',
)

parser.add_argument(
    "--force_offload123",
    default=True,
    help='Argument 6, input `force_offload` for node "WanVideo ClipVision Encode" id 87 (autogenerated)',
)

parser.add_argument(
    "--tiles124",
    default=0,
    help='Argument 9, input `tiles` for node "WanVideo ClipVision Encode" id 87 (autogenerated)',
)

parser.add_argument(
    "--ratio125",
    default=0.5,
    help='Argument 10, input `ratio` for node "WanVideo ClipVision Encode" id 87 (autogenerated)',
)

parser.add_argument(
    "--expression126",
    default="ceil(a/b)",
    help='Argument 0, input `expression` for node "Calculate FWS" id 144 (autogenerated)',
)

parser.add_argument(
    "--force_offload127",
    default=False,
    help='Argument 4, input `force_offload` for node "WanVideo Animate Embeds" id 88 (autogenerated)',
)

parser.add_argument(
    "--colormatch128",
    default="disabled",
    help='Argument 6, input `colormatch` for node "WanVideo Animate Embeds" id 88 (autogenerated)',
)

parser.add_argument(
    "--pose_strength129",
    default=1,
    help='Argument 7, input `pose_strength` for node "WanVideo Animate Embeds" id 88 (autogenerated)',
)

parser.add_argument(
    "--face_strength130",
    default=1,
    help='Argument 8, input `face_strength` for node "WanVideo Animate Embeds" id 88 (autogenerated)',
)

parser.add_argument(
    "--tiled_vae131",
    default=False,
    help='Argument 15, input `tiled_vae` for node "WanVideo Animate Embeds" id 88 (autogenerated)',
)

parser.add_argument(
    "--steps132",
    default=6,
    help='Argument 2, input `steps` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--cfg133",
    default=1,
    help='Argument 3, input `cfg` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--shift134",
    default=3,
    help='Argument 4, input `shift` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--seed135",
    default=777,
    help='Argument 5, input `seed` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--force_offload136",
    default=False,
    help='Argument 6, input `force_offload` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--scheduler137",
    default="lcm",
    help='Argument 7, input `scheduler` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--riflex_freq_index138",
    default=0,
    help='Argument 8, input `riflex_freq_index` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--denoise_strength139",
    default=1,
    help='Argument 11, input `denoise_strength` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--batched_cfg140",
    default="",
    help='Argument 16, input `batched_cfg` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--rope_function141",
    default="comfy",
    help='Argument 18, input `rope_function` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--start_step142",
    default=0,
    help='Argument 27, input `start_step` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--end_step143",
    default=-1,
    help='Argument 28, input `end_step` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--add_noise_to_samples144",
    default=False,
    help='Argument 29, input `add_noise_to_samples` for node "WanVideo Sampler" id 90 (autogenerated)',
)

parser.add_argument(
    "--enable_vae_tiling145",
    default=False,
    help='Argument 2, input `enable_vae_tiling` for node "WanVideo Decode" id 91 (autogenerated)',
)

parser.add_argument(
    "--tile_x146",
    default=272,
    help='Argument 3, input `tile_x` for node "WanVideo Decode" id 91 (autogenerated)',
)

parser.add_argument(
    "--tile_y147",
    default=272,
    help='Argument 4, input `tile_y` for node "WanVideo Decode" id 91 (autogenerated)',
)

parser.add_argument(
    "--tile_stride_x148",
    default=144,
    help='Argument 5, input `tile_stride_x` for node "WanVideo Decode" id 91 (autogenerated)',
)

parser.add_argument(
    "--tile_stride_y149",
    default=128,
    help='Argument 6, input `tile_stride_y` for node "WanVideo Decode" id 91 (autogenerated)',
)

parser.add_argument(
    "--normalization150",
    default="default",
    help='Argument 7, input `normalization` for node "WanVideo Decode" id 91 (autogenerated)',
)

parser.add_argument(
    "--loop_count151",
    default=0,
    help='Argument 2, input `loop_count` for node "Video Combine 🎥🅥🅗🅢" id 106 (autogenerated)',
)

parser.add_argument(
    "--filename_prefix152",
    default="Wanimate",
    help='Argument 3, input `filename_prefix` for node "Video Combine 🎥🅥🅗🅢" id 106 (autogenerated)',
)

parser.add_argument(
    "--format153",
    default="video/h264-mp4",
    help='Argument 4, input `format` for node "Video Combine 🎥🅥🅗🅢" id 106 (autogenerated)',
)

parser.add_argument(
    "--pingpong154",
    default=False,
    help='Argument 5, input `pingpong` for node "Video Combine 🎥🅥🅗🅢" id 106 (autogenerated)',
)

parser.add_argument(
    "--save_output155",
    default=True,
    help='Argument 6, input `save_output` for node "Video Combine 🎥🅥🅗🅢" id 106 (autogenerated)',
)

parser.add_argument(
    "--frame_rate156",
    default=16,
    help='Argument 1, input `frame_rate` for node "Video Combine 🎥🅥🅗🅢" id 141 (autogenerated)',
)

parser.add_argument(
    "--loop_count157",
    default=0,
    help='Argument 2, input `loop_count` for node "Video Combine 🎥🅥🅗🅢" id 141 (autogenerated)',
)

parser.add_argument(
    "--filename_prefix158",
    default="WanVideo2_1_T2V",
    help='Argument 3, input `filename_prefix` for node "Video Combine 🎥🅥🅗🅢" id 141 (autogenerated)',
)

parser.add_argument(
    "--format159",
    default="video/h264-mp4",
    help='Argument 4, input `format` for node "Video Combine 🎥🅥🅗🅢" id 141 (autogenerated)',
)

parser.add_argument(
    "--pingpong160",
    default=False,
    help='Argument 5, input `pingpong` for node "Video Combine 🎥🅥🅗🅢" id 141 (autogenerated)',
)

parser.add_argument(
    "--save_output161",
    default=False,
    help='Argument 6, input `save_output` for node "Video Combine 🎥🅥🅗🅢" id 141 (autogenerated)',
)

parser.add_argument(
    "--ckpt_name162",
    default="rife49.pth",
    help='Argument 0, input `ckpt_name` for node "RIFE VFI" id 171 (autogenerated)',
)

parser.add_argument(
    "--fast_mode163",
    default=False,
    help='Argument 4, input `fast_mode` for node "RIFE VFI" id 171 (autogenerated)',
)

parser.add_argument(
    "--ensemble164",
    default=True,
    help='Argument 5, input `ensemble` for node "RIFE VFI" id 171 (autogenerated)',
)

parser.add_argument(
    "--scale_factor165",
    default=1,
    help='Argument 6, input `scale_factor` for node "RIFE VFI" id 171 (autogenerated)',
)

parser.add_argument(
    "--expression166",
    default="a*b",
    help='Argument 0, input `expression` for node "Calculate new FR" id 189 (autogenerated)',
)

parser.add_argument(
    "--loop_count167",
    default=0,
    help='Argument 2, input `loop_count` for node "Video Combine 🎥🅥🅗🅢" id 178 (autogenerated)',
)

parser.add_argument(
    "--filename_prefix168",
    default="Wanimate_Interpolated",
    help='Argument 3, input `filename_prefix` for node "Video Combine 🎥🅥🅗🅢" id 178 (autogenerated)',
)

parser.add_argument(
    "--format169",
    default="video/h264-mp4",
    help='Argument 4, input `format` for node "Video Combine 🎥🅥🅗🅢" id 178 (autogenerated)',
)

parser.add_argument(
    "--pingpong170",
    default=False,
    help='Argument 5, input `pingpong` for node "Video Combine 🎥🅥🅗🅢" id 178 (autogenerated)',
)

parser.add_argument(
    "--save_output171",
    default=True,
    help='Argument 6, input `save_output` for node "Video Combine 🎥🅥🅗🅢" id 178 (autogenerated)',
)

parser.add_argument(
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)


comfy_args = [sys.argv[0]]
if __name__ == "__main__" and "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output is not None and args.output == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()

PROMPT_DATA = json.loads(
    '{"9": {"inputs": {"frame_count": 1, "revert": true, "frames": ["33", 0]}, "class_type": "GenerateFramesByCount", "_meta": {"title": "Generate Frames By Count"}}, "10": {"inputs": {"images": ["9", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Check Video Crop/Size"}}, "13": {"inputs": {"video": "video.mp4", "force_rate": ["30", 0], "custom_width": 0, "custom_height": 0, "frame_load_cap": ["21", 0], "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan"}, "class_type": "VHS_LoadVideo", "_meta": {"title": "Load Video (Upload) \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "15": {"inputs": {"images": ["63", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "20": {"inputs": {"text": "109", "anything": ["13", 1]}, "class_type": "easy showAnything", "_meta": {"title": "Check frame count"}}, "21": {"inputs": {"expression": "(a*b)+1", "a": ["28", 0], "b": ["30", 0]}, "class_type": "MathExpression|pysssss", "_meta": {"title": "Calculate Frames"}}, "28": {"inputs": {"value": ["228", 2]}, "class_type": "FloatConstant", "_meta": {"title": "\\ud83d\\udd22 Length (Seconds)"}}, "29": {"inputs": {"value": 2}, "class_type": "INTConstant", "_meta": {"title": "\\ud83d\\udd00 Interpolation factor"}}, "30": {"inputs": {"value": 16}, "class_type": "PrimitiveFloat", "_meta": {"title": "\\ud83c\\udfa6 Framerate"}}, "32": {"inputs": {"value": 720}, "class_type": "INTConstant", "_meta": {"title": "\\ud83d\\udd1b Resolution"}}, "33": {"inputs": {"width": ["32", 0], "height": 10000, "upscale_method": "lanczos", "keep_proportion": "resize", "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 16, "device": "cpu", "per_batch": 0, "image": ["13", 0]}, "class_type": "ImageResizeKJv2", "_meta": {"title": "Resize Image v2"}}, "36": {"inputs": {"model": "sam2_hiera_base_plus.safetensors", "segmentor": "video", "device": "cuda", "precision": "fp16"}, "class_type": "DownloadAndLoadSAM2Model", "_meta": {"title": "(Down)Load SAM2Model"}}, "37": {"inputs": {"keep_model_loaded": false, "individual_objects": false, "sam2_model": ["36", 0], "image": ["33", 0], "coordinates_positive": ["255", 0]}, "class_type": "Sam2Segmentation", "_meta": {"title": "Sam2Segmentation"}}, "38": {"inputs": {"expand": 10, "tapered_corners": true, "mask": ["37", 0]}, "class_type": "GrowMask", "_meta": {"title": "GrowMask"}}, "39": {"inputs": {"block_size": 32, "device": "cpu", "masks": ["38", 0]}, "class_type": "BlockifyMask", "_meta": {"title": "Blockify Mask"}}, "43": {"inputs": {"color": "0, 0, 0", "device": "cpu", "image": ["33", 0], "mask": ["39", 0]}, "class_type": "DrawMaskOnImage", "_meta": {"title": "Draw Mask On Image"}}, "44": {"inputs": {"frame_rate": 16, "loop_count": 0, "filename_prefix": "WanVideo2_1_T2V", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": true, "trim_to_audio": false, "pingpong": false, "save_output": true, "images": ["43", 0]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "52": {"inputs": {"frame_rate": 16, "loop_count": 0, "filename_prefix": "WanVideo2_1_T2V", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": true, "trim_to_audio": false, "pingpong": false, "save_output": false, "images": ["53", 0]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "53": {"inputs": {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable", "resolution": ["58", 0], "bbox_detector": "yolox_l.torchscript.pt", "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "scale_stick_for_xinsr_cn": "disable", "image": ["33", 0]}, "class_type": "DWPreprocessor", "_meta": {"title": "DWPose Estimator"}}, "56": {"inputs": {"base_resolution": 512, "padding": 0, "min_crop_resolution": 128, "max_crop_resolution": 512, "image": ["33", 0], "mask": ["260", 0]}, "class_type": "ImageCropByMaskAndResize", "_meta": {"title": "Image Crop By Mask And Resize"}}, "58": {"inputs": {"image_gen_width": ["33", 1], "image_gen_height": ["33", 2], "resize_mode": "Just Resize", "original_image": ["33", 0]}, "class_type": "PixelPerfectResolution", "_meta": {"title": "Pixel Perfect Resolution"}}, "63": {"inputs": {"width": ["33", 1], "height": 10000, "upscale_method": "lanczos", "keep_proportion": "resize", "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 16, "device": "cpu", "per_batch": 0, "image": ["64", 0]}, "class_type": "ImageResizeKJv2", "_meta": {"title": "Resize Image v2"}}, "64": {"inputs": {"image": "image.jpeg"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "75": {"inputs": {"model": "Wan2_2-Animate-14B_fp8_scaled_e5m2_KJ_v2.safetensors", "base_precision": "fp16_fast", "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sageattn", "rms_norm_function": "default", "compile_args": ["82", 0], "block_swap_args": ["238", 0], "lora": ["76", 0]}, "class_type": "WanVideoModelLoader", "_meta": {"title": "WanVideo Model Loader"}}, "76": {"inputs": {"lora_0": "WanAnimate_relight_lora_fp16.safetensors", "strength_0": 1, "lora_1": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_1": 1.2, "lora_2": "none", "strength_2": 1, "lora_3": "none", "strength_3": 1, "lora_4": "none", "strength_4": 1, "low_mem_load": false, "merge_loras": false}, "class_type": "WanVideoLoraSelectMulti", "_meta": {"title": "WanVideo Lora Select Multi"}}, "80": {"inputs": {"clip_name": "clip_vision_h.safetensors"}, "class_type": "CLIPVisionLoader", "_meta": {"title": "Load CLIP Vision"}}, "81": {"inputs": {"model_name": "Wan2_1_VAE_bf16.safetensors", "precision": "fp16"}, "class_type": "WanVideoVAELoader", "_meta": {"title": "WanVideo VAE Loader"}}, "82": {"inputs": {"backend": "inductor", "fullgraph": false, "mode": "default", "dynamic": false, "dynamo_cache_size_limit": 64, "compile_transformer_blocks_only": true, "dynamo_recompile_limit": 128}, "class_type": "WanVideoTorchCompileSettings", "_meta": {"title": "WanVideo Torch Compile Settings"}}, "87": {"inputs": {"strength_1": 1, "strength_2": 1, "crop": "center", "combine_embeds": "average", "force_offload": true, "tiles": 0, "ratio": 0.5, "clip_vision": ["80", 0], "image_1": ["63", 0]}, "class_type": "WanVideoClipVisionEncode", "_meta": {"title": "WanVideo ClipVision Encode"}}, "88": {"inputs": {"width": ["33", 1], "height": ["33", 2], "num_frames": ["13", 1], "force_offload": false, "frame_window_size": ["144", 0], "colormatch": "disabled", "pose_strength": 1, "face_strength": 1, "tiled_vae": false, "vae": ["81", 0], "clip_embeds": ["87", 0], "ref_images": ["63", 0], "pose_images": ["53", 0], "face_images": ["56", 0], "bg_images": ["43", 0], "mask": ["39", 0]}, "class_type": "WanVideoAnimateEmbeds", "_meta": {"title": "WanVideo Animate Embeds"}}, "90": {"inputs": {"steps": 6, "cfg": 1, "shift": 3, "seed": 777, "force_offload": false, "scheduler": "lcm", "riflex_freq_index": 0, "denoise_strength": 1, "batched_cfg": "", "rope_function": "comfy", "start_step": 0, "end_step": -1, "add_noise_to_samples": false, "model": ["75", 0], "image_embeds": ["88", 0], "text_embeds": ["125", 0]}, "class_type": "WanVideoSampler", "_meta": {"title": "WanVideo Sampler"}}, "91": {"inputs": {"enable_vae_tiling": false, "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128, "normalization": "default", "vae": ["81", 0], "samples": ["90", 0]}, "class_type": "WanVideoDecode", "_meta": {"title": "WanVideo Decode"}}, "92": {"inputs": {"image": ["91", 0]}, "class_type": "GetImageSizeAndCount", "_meta": {"title": "Get Image Size & Count"}}, "106": {"inputs": {"frame_rate": ["30", 0], "loop_count": 0, "filename_prefix": "Wanimate", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": true, "trim_to_audio": false, "pingpong": false, "save_output": true, "images": ["92", 0], "audio": ["13", 2]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "119": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader", "_meta": {"title": "Load CLIP"}}, "122": {"inputs": {"text": "\\u8fc7\\u66dd\\uff0c\\u9759\\u6001\\uff0c\\u7ec6\\u8282\\u6a21\\u7cca\\u4e0d\\u6e05\\uff0c\\u5b57\\u5e55\\uff0c\\u98ce\\u683c\\uff0c\\u4f5c\\u54c1\\uff0c\\u753b\\u4f5c\\uff0c\\u753b\\u9762\\uff0c\\u9759\\u6b62\\uff0c\\u6574\\u4f53\\u53d1\\u7070\\uff0c\\u6700\\u5dee\\u8d28\\u91cf\\uff0c\\u4f4e\\u8d28\\u91cf\\uff0cJPEG\\u538b\\u7f29\\u6b8b\\u7559\\uff0c\\u4e11\\u964b\\u7684\\uff0c\\u6b8b\\u7f3a\\u7684\\uff0c\\u591a\\u4f59\\u7684\\u624b\\u6307\\uff0c\\u753b\\u5f97\\u4e0d\\u597d\\u7684\\u624b\\u90e8\\uff0c\\u753b\\u5f97\\u4e0d\\u597d\\u7684\\u8138\\u90e8\\uff0c\\u7578\\u5f62\\u7684\\uff0c\\u6bc1\\u5bb9\\u7684\\uff0c\\u5f62\\u6001\\u7578\\u5f62\\u7684\\u80a2\\u4f53\\uff0c\\u624b\\u6307\\u878d\\u5408\\uff0c\\u9759\\u6b62\\u4e0d\\u52a8\\u7684\\u753b\\u9762\\uff0c\\u6742\\u4e71\\u7684\\u80cc\\u666f\\uff0c\\u4e09\\u6761\\u817f\\uff0c\\u80cc\\u666f\\u4eba\\u5f88\\u591a\\uff0c\\u5012\\u7740\\u8d70, tattoo,", "clip": ["119", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "124": {"inputs": {"text": "", "clip": ["119", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "125": {"inputs": {"positive": ["124", 0], "negative": ["122", 0]}, "class_type": "WanVideoTextEmbedBridge", "_meta": {"title": "WanVideo TextEmbed Bridge"}}, "141": {"inputs": {"frame_rate": 16, "loop_count": 0, "filename_prefix": "WanVideo2_1_T2V", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": false, "trim_to_audio": false, "pingpong": false, "save_output": false, "images": ["56", 0]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "144": {"inputs": {"expression": "ceil(a/b)", "a": ["21", 0], "b": ["161", 0]}, "class_type": "MathExpression|pysssss", "_meta": {"title": "Calculate FWS"}}, "147": {"inputs": {"text": "110", "anything": ["144", 0]}, "class_type": "easy showAnything", "_meta": {"title": "Window Frame Size"}}, "148": {"inputs": {"text": "110", "anything": ["21", 0]}, "class_type": "easy showAnything", "_meta": {"title": "Frame Count"}}, "161": {"inputs": {"value": 1}, "class_type": "INTConstant", "_meta": {"title": "\\ud83d\\udd23 Divide FWS"}}, "171": {"inputs": {"ckpt_name": "rife49.pth", "clear_cache_after_n_frames": ["176", 0], "multiplier": ["29", 0], "fast_mode": false, "ensemble": true, "scale_factor": 1, "frames": ["92", 0]}, "class_type": "RIFE VFI", "_meta": {"title": "RIFE VFI"}}, "176": {"inputs": {"a": ["30", 0]}, "class_type": "CM_FloatToInt", "_meta": {"title": "FloatToInt"}}, "178": {"inputs": {"frame_rate": ["189", 1], "loop_count": 0, "filename_prefix": "Wanimate_Interpolated", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": true, "trim_to_audio": false, "pingpong": false, "save_output": true, "images": ["171", 0], "audio": ["13", 2]}, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "189": {"inputs": {"expression": "a*b", "a": ["30", 0], "b": ["29", 0]}, "class_type": "MathExpression|pysssss", "_meta": {"title": "Calculate new FR"}}, "208": {"inputs": {"images": ["53", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "222": {"inputs": {"video": "video.mp4", "force_rate": 0, "custom_width": 0, "custom_height": 0, "frame_load_cap": 65, "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan"}, "class_type": "VHS_LoadVideo", "_meta": {"title": "Load Video (Upload) \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "228": {"inputs": {"video_info": ["222", 3]}, "class_type": "VHS_VideoInfoSource", "_meta": {"title": "Video Info (Source) \\ud83c\\udfa5\\ud83c\\udd65\\ud83c\\udd57\\ud83c\\udd62"}}, "238": {"inputs": {"blocks_to_swap": 40, "offload_img_emb": false, "offload_txt_emb": false, "use_non_blocking": false, "vace_blocks_to_swap": 0, "prefetch_blocks": 0, "block_swap_debug": false}, "class_type": "WanVideoBlockSwap", "_meta": {"title": "WanVideo Block Swap"}}, "255": {"inputs": {"person_index": ["263", 0], "top_k_pose": 5, "top_k_face": 1, "top_k_left_hand": 1, "top_k_right_hand": 1, "pose": ["53", 1]}, "class_type": "PoseToSAMPoints", "_meta": {"title": "PoseToSAMPoints"}}, "260": {"inputs": {"person_index": ["263", 0], "pose_kps": ["53", 1]}, "class_type": "FaceMaskFromPoseKeypoints", "_meta": {"title": "Face Mask From Pose Keypoints"}}, "263": {"inputs": {"value": 0}, "class_type": "INTConstant", "_meta": {"title": "Person Index"}}}'
)


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    if has_manager:
        try:
            import manager_core as manager
        except ImportError:
            print("Could not import manager_core, proceeding without it.")
            return
        else:
            if hasattr(manager, "get_config"):
                print("Patching manager_core.get_config to enforce offline mode.")
                try:
                    get_config = manager.get_config

                    def _get_config(*args, **kwargs):
                        config = get_config(*args, **kwargs)
                        config["network_mode"] = "offline"
                        return config

                    manager.get_config = _get_config
                except Exception as e:
                    print("Failed to patch manager_core.get_config:", e)

    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def inner():
        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        await init_extra_nodes(init_custom_nodes=True)

    loop.run_until_complete(inner())


_custom_nodes_imported = False
_custom_path_added = False


def main(*func_args, **func_kwargs):
    global args, _custom_nodes_imported, _custom_path_added
    if __name__ == "__main__":
        if args is None:
            args = parser.parse_args()
    else:
        defaults = dict(
            (arg, parser.get_default(arg))
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata"]
            + [
                "value1",
                "value2",
                "value3",
                "model4",
                "segmentor5",
                "device6",
                "precision7",
                "image8",
                "lora_09",
                "strength_010",
                "lora_111",
                "strength_112",
                "lora_213",
                "strength_214",
                "lora_315",
                "strength_316",
                "lora_417",
                "strength_418",
                "low_mem_load19",
                "merge_loras20",
                "clip_name21",
                "model_name22",
                "precision23",
                "backend24",
                "fullgraph25",
                "mode26",
                "dynamic27",
                "dynamo_cache_size_limit28",
                "compile_transformer_blocks_only29",
                "dynamo_recompile_limit30",
                "clip_name31",
                "type32",
                "device33",
                "text34",
                "text35",
                "value36",
                "video37",
                "force_rate38",
                "custom_width39",
                "custom_height40",
                "frame_load_cap41",
                "skip_first_frames42",
                "select_every_nth43",
                "format44",
                "blocks_to_swap45",
                "offload_img_emb46",
                "offload_txt_emb47",
                "use_non_blocking48",
                "vace_blocks_to_swap49",
                "prefetch_blocks50",
                "block_swap_debug51",
                "value52",
                "expression53",
                "video54",
                "custom_width55",
                "custom_height56",
                "skip_first_frames57",
                "select_every_nth58",
                "format59",
                "height60",
                "upscale_method61",
                "keep_proportion62",
                "pad_color63",
                "crop_position64",
                "divisible_by65",
                "device66",
                "per_batch67",
                "frame_count68",
                "revert69",
                "height70",
                "upscale_method71",
                "keep_proportion72",
                "pad_color73",
                "crop_position74",
                "divisible_by75",
                "device76",
                "per_batch77",
                "resize_mode78",
                "detect_hand79",
                "detect_body80",
                "detect_face81",
                "bbox_detector82",
                "pose_estimator83",
                "scale_stick_for_xinsr_cn84",
                "top_k_pose85",
                "top_k_face86",
                "top_k_left_hand87",
                "top_k_right_hand88",
                "keep_model_loaded89",
                "individual_objects90",
                "expand91",
                "tapered_corners92",
                "block_size93",
                "device94",
                "color95",
                "device96",
                "frame_rate97",
                "loop_count98",
                "filename_prefix99",
                "format100",
                "pingpong101",
                "save_output102",
                "frame_rate103",
                "loop_count104",
                "filename_prefix105",
                "format106",
                "pingpong107",
                "save_output108",
                "base_resolution109",
                "padding110",
                "min_crop_resolution111",
                "max_crop_resolution112",
                "model113",
                "base_precision114",
                "quantization115",
                "load_device116",
                "attention_mode117",
                "rms_norm_function118",
                "strength_1119",
                "strength_2120",
                "crop121",
                "combine_embeds122",
                "force_offload123",
                "tiles124",
                "ratio125",
                "expression126",
                "force_offload127",
                "colormatch128",
                "pose_strength129",
                "face_strength130",
                "tiled_vae131",
                "steps132",
                "cfg133",
                "shift134",
                "seed135",
                "force_offload136",
                "scheduler137",
                "riflex_freq_index138",
                "denoise_strength139",
                "batched_cfg140",
                "rope_function141",
                "start_step142",
                "end_step143",
                "add_noise_to_samples144",
                "enable_vae_tiling145",
                "tile_x146",
                "tile_y147",
                "tile_stride_x148",
                "tile_stride_y149",
                "normalization150",
                "loop_count151",
                "filename_prefix152",
                "format153",
                "pingpong154",
                "save_output155",
                "frame_rate156",
                "loop_count157",
                "filename_prefix158",
                "format159",
                "pingpong160",
                "save_output161",
                "ckpt_name162",
                "fast_mode163",
                "ensemble164",
                "scale_factor165",
                "expression166",
                "loop_count167",
                "filename_prefix168",
                "format169",
                "pingpong170",
                "save_output171",
            ]
        )

        all_args = dict()
        all_args.update(defaults)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    with ctx:
        if not _custom_path_added:
            add_comfyui_directory_to_sys_path()
            add_extra_model_paths()

            _custom_path_added = True

        if not _custom_nodes_imported:
            import_custom_nodes()

            _custom_nodes_imported = True

        from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode(), ctx:
        intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
        intconstant_29 = intconstant.get_value(value=parse_arg(args.value1))

        primitivefloat = NODE_CLASS_MAPPINGS["PrimitiveFloat"]()
        primitivefloat_30 = primitivefloat.EXECUTE_NORMALIZED(
            value=parse_arg(args.value2)
        )

        intconstant_32 = intconstant.get_value(value=parse_arg(args.value3))

        downloadandloadsam2model = NODE_CLASS_MAPPINGS["DownloadAndLoadSAM2Model"]()
        downloadandloadsam2model_36 = downloadandloadsam2model.loadmodel(
            model=parse_arg(args.model4),
            segmentor=parse_arg(args.segmentor5),
            device=parse_arg(args.device6),
            precision=parse_arg(args.precision7),
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_64 = loadimage.load_image(image=parse_arg(args.image8))

        wanvideoloraselectmulti = NODE_CLASS_MAPPINGS["WanVideoLoraSelectMulti"]()
        wanvideoloraselectmulti_76 = wanvideoloraselectmulti.getlorapath(
            lora_0=parse_arg(args.lora_09),
            strength_0=parse_arg(args.strength_010),
            lora_1=parse_arg(args.lora_111),
            strength_1=parse_arg(args.strength_112),
            lora_2=parse_arg(args.lora_213),
            strength_2=parse_arg(args.strength_214),
            lora_3=parse_arg(args.lora_315),
            strength_3=parse_arg(args.strength_316),
            lora_4=parse_arg(args.lora_417),
            strength_4=parse_arg(args.strength_418),
            low_mem_load=parse_arg(args.low_mem_load19),
            merge_loras=parse_arg(args.merge_loras20),
        )

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_80 = clipvisionloader.load_clip(
            clip_name=parse_arg(args.clip_name21)
        )

        wanvideovaeloader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        wanvideovaeloader_81 = wanvideovaeloader.loadmodel(
            model_name=parse_arg(args.model_name22),
            precision=parse_arg(args.precision23),
        )

        wanvideotorchcompilesettings = NODE_CLASS_MAPPINGS[
            "WanVideoTorchCompileSettings"
        ]()
        wanvideotorchcompilesettings_82 = wanvideotorchcompilesettings.set_args(
            backend=parse_arg(args.backend24),
            fullgraph=parse_arg(args.fullgraph25),
            mode=parse_arg(args.mode26),
            dynamic=parse_arg(args.dynamic27),
            dynamo_cache_size_limit=parse_arg(args.dynamo_cache_size_limit28),
            compile_transformer_blocks_only=parse_arg(
                args.compile_transformer_blocks_only29
            ),
            dynamo_recompile_limit=parse_arg(args.dynamo_recompile_limit30),
        )

        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_119 = cliploader.load_clip(
            clip_name=parse_arg(args.clip_name31),
            type=parse_arg(args.type32),
            device=parse_arg(args.device33),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_122 = cliptextencode.encode(
            text=parse_arg(args.text34), clip=get_value_at_index(cliploader_119, 0)
        )

        cliptextencode_124 = cliptextencode.encode(
            text=parse_arg(args.text35), clip=get_value_at_index(cliploader_119, 0)
        )

        intconstant_161 = intconstant.get_value(value=parse_arg(args.value36))

        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
        vhs_loadvideo_222 = vhs_loadvideo.load_video(
            video=parse_arg(args.video37),
            force_rate=parse_arg(args.force_rate38),
            custom_width=parse_arg(args.custom_width39),
            custom_height=parse_arg(args.custom_height40),
            frame_load_cap=parse_arg(args.frame_load_cap41),
            skip_first_frames=parse_arg(args.skip_first_frames42),
            select_every_nth=parse_arg(args.select_every_nth43),
            format=parse_arg(args.format44),
        )

        wanvideoblockswap = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
        wanvideoblockswap_238 = wanvideoblockswap.setargs(
            blocks_to_swap=parse_arg(args.blocks_to_swap45),
            offload_img_emb=parse_arg(args.offload_img_emb46),
            offload_txt_emb=parse_arg(args.offload_txt_emb47),
            use_non_blocking=parse_arg(args.use_non_blocking48),
            vace_blocks_to_swap=parse_arg(args.vace_blocks_to_swap49),
            prefetch_blocks=parse_arg(args.prefetch_blocks50),
            block_swap_debug=parse_arg(args.block_swap_debug51),
        )

        intconstant_263 = intconstant.get_value(value=parse_arg(args.value52))

        vhs_videoinfosource = NODE_CLASS_MAPPINGS["VHS_VideoInfoSource"]()
        floatconstant = NODE_CLASS_MAPPINGS["FloatConstant"]()
        mathexpressionpysssss = NODE_CLASS_MAPPINGS["MathExpression|pysssss"]()
        imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        generateframesbycount = NODE_CLASS_MAPPINGS["GenerateFramesByCount"]()
        pixelperfectresolution = NODE_CLASS_MAPPINGS["PixelPerfectResolution"]()
        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        posetosampoints = NODE_CLASS_MAPPINGS["PoseToSAMPoints"]()
        sam2segmentation = NODE_CLASS_MAPPINGS["Sam2Segmentation"]()
        growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        blockifymask = NODE_CLASS_MAPPINGS["BlockifyMask"]()
        drawmaskonimage = NODE_CLASS_MAPPINGS["DrawMaskOnImage"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        facemaskfromposekeypoints = NODE_CLASS_MAPPINGS["FaceMaskFromPoseKeypoints"]()
        imagecropbymaskandresize = NODE_CLASS_MAPPINGS["ImageCropByMaskAndResize"]()
        wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        wanvideoclipvisionencode = NODE_CLASS_MAPPINGS["WanVideoClipVisionEncode"]()
        wanvideoanimateembeds = NODE_CLASS_MAPPINGS["WanVideoAnimateEmbeds"]()
        wanvideotextembedbridge = NODE_CLASS_MAPPINGS["WanVideoTextEmbedBridge"]()
        wanvideosampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        wanvideodecode = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
        getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
        cm_floattoint = NODE_CLASS_MAPPINGS["CM_FloatToInt"]()
        rife_vfi = NODE_CLASS_MAPPINGS["RIFE VFI"]()
        for q in range(args.queue_size):
            vhs_videoinfosource_228 = vhs_videoinfosource.get_video_info(
                video_info=get_value_at_index(vhs_loadvideo_222, 3)
            )

            floatconstant_28 = floatconstant.get_value(
                value=get_value_at_index(vhs_videoinfosource_228, 2)
            )

            mathexpressionpysssss_21 = mathexpressionpysssss.evaluate(
                expression=parse_arg(args.expression53),
                a=get_value_at_index(floatconstant_28, 0),
                b=get_value_at_index(primitivefloat_30, 0),
                prompt=PROMPT_DATA,
            )

            vhs_loadvideo_13 = vhs_loadvideo.load_video(
                video=parse_arg(args.video54),
                force_rate=get_value_at_index(primitivefloat_30, 0),
                custom_width=parse_arg(args.custom_width55),
                custom_height=parse_arg(args.custom_height56),
                frame_load_cap=get_value_at_index(mathexpressionpysssss_21, 0),
                skip_first_frames=parse_arg(args.skip_first_frames57),
                select_every_nth=parse_arg(args.select_every_nth58),
                format=parse_arg(args.format59),
            )

            imageresizekjv2_33 = imageresizekjv2.resize(
                width=get_value_at_index(intconstant_32, 0),
                height=parse_arg(args.height60),
                upscale_method=parse_arg(args.upscale_method61),
                keep_proportion=parse_arg(args.keep_proportion62),
                pad_color=parse_arg(args.pad_color63),
                crop_position=parse_arg(args.crop_position64),
                divisible_by=parse_arg(args.divisible_by65),
                device=parse_arg(args.device66),
                per_batch=parse_arg(args.per_batch67),
                image=get_value_at_index(vhs_loadvideo_13, 0),
                unique_id=1846348952956490158,
            )

            generateframesbycount_9 = generateframesbycount.r(
                frame_count=parse_arg(args.frame_count68),
                revert=parse_arg(args.revert69),
                frames=get_value_at_index(imageresizekjv2_33, 0),
            )

            imageresizekjv2_63 = imageresizekjv2.resize(
                width=get_value_at_index(imageresizekjv2_33, 1),
                height=parse_arg(args.height70),
                upscale_method=parse_arg(args.upscale_method71),
                keep_proportion=parse_arg(args.keep_proportion72),
                pad_color=parse_arg(args.pad_color73),
                crop_position=parse_arg(args.crop_position74),
                divisible_by=parse_arg(args.divisible_by75),
                device=parse_arg(args.device76),
                per_batch=parse_arg(args.per_batch77),
                image=get_value_at_index(loadimage_64, 0),
                unique_id=13528543081259825256,
            )

            pixelperfectresolution_58 = pixelperfectresolution.execute(
                image_gen_width=get_value_at_index(imageresizekjv2_33, 1),
                image_gen_height=get_value_at_index(imageresizekjv2_33, 2),
                resize_mode=parse_arg(args.resize_mode78),
                original_image=get_value_at_index(imageresizekjv2_33, 0),
            )

            dwpreprocessor_53 = dwpreprocessor.estimate_pose(
                detect_hand=parse_arg(args.detect_hand79),
                detect_body=parse_arg(args.detect_body80),
                detect_face=parse_arg(args.detect_face81),
                resolution=get_value_at_index(pixelperfectresolution_58, 0),
                bbox_detector=parse_arg(args.bbox_detector82),
                pose_estimator=parse_arg(args.pose_estimator83),
                scale_stick_for_xinsr_cn=parse_arg(args.scale_stick_for_xinsr_cn84),
                image=get_value_at_index(imageresizekjv2_33, 0),
            )

            posetosampoints_255 = posetosampoints.convert(
                person_index=get_value_at_index(intconstant_263, 0),
                top_k_pose=parse_arg(args.top_k_pose85),
                top_k_face=parse_arg(args.top_k_face86),
                top_k_left_hand=parse_arg(args.top_k_left_hand87),
                top_k_right_hand=parse_arg(args.top_k_right_hand88),
                pose=get_value_at_index(dwpreprocessor_53, 1),
            )

            sam2segmentation_37 = sam2segmentation.segment(
                keep_model_loaded=parse_arg(args.keep_model_loaded89),
                individual_objects=parse_arg(args.individual_objects90),
                sam2_model=get_value_at_index(downloadandloadsam2model_36, 0),
                image=get_value_at_index(imageresizekjv2_33, 0),
                coordinates_positive=get_value_at_index(posetosampoints_255, 0),
            )

            growmask_38 = growmask.expand_mask(
                expand=parse_arg(args.expand91),
                tapered_corners=parse_arg(args.tapered_corners92),
                mask=get_value_at_index(sam2segmentation_37, 0),
            )

            blockifymask_39 = blockifymask.process(
                block_size=parse_arg(args.block_size93),
                device=parse_arg(args.device94),
                masks=get_value_at_index(growmask_38, 0),
            )

            drawmaskonimage_43 = drawmaskonimage.apply(
                color=parse_arg(args.color95),
                device=parse_arg(args.device96),
                image=get_value_at_index(imageresizekjv2_33, 0),
                mask=get_value_at_index(blockifymask_39, 0),
            )

            vhs_videocombine_44 = vhs_videocombine.combine_video(
                frame_rate=parse_arg(args.frame_rate97),
                loop_count=parse_arg(args.loop_count98),
                filename_prefix=parse_arg(args.filename_prefix99),
                format=parse_arg(args.format100),
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=parse_arg(args.pingpong101),
                save_output=parse_arg(args.save_output102),
                images=get_value_at_index(drawmaskonimage_43, 0),
            )

            vhs_videocombine_52 = vhs_videocombine.combine_video(
                frame_rate=parse_arg(args.frame_rate103),
                loop_count=parse_arg(args.loop_count104),
                filename_prefix=parse_arg(args.filename_prefix105),
                format=parse_arg(args.format106),
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=parse_arg(args.pingpong107),
                save_output=parse_arg(args.save_output108),
                images=get_value_at_index(dwpreprocessor_53, 0),
            )

            facemaskfromposekeypoints_260 = facemaskfromposekeypoints.createmask(
                person_index=get_value_at_index(intconstant_263, 0),
                pose_kps=get_value_at_index(dwpreprocessor_53, 1),
            )

            imagecropbymaskandresize_56 = imagecropbymaskandresize.crop(
                base_resolution=parse_arg(args.base_resolution109),
                padding=parse_arg(args.padding110),
                min_crop_resolution=parse_arg(args.min_crop_resolution111),
                max_crop_resolution=parse_arg(args.max_crop_resolution112),
                image=get_value_at_index(imageresizekjv2_33, 0),
                mask=get_value_at_index(facemaskfromposekeypoints_260, 0),
            )

            wanvideomodelloader_75 = wanvideomodelloader.loadmodel(
                model=parse_arg(args.model113),
                base_precision=parse_arg(args.base_precision114),
                quantization=parse_arg(args.quantization115),
                load_device=parse_arg(args.load_device116),
                attention_mode=parse_arg(args.attention_mode117),
                rms_norm_function=parse_arg(args.rms_norm_function118),
                compile_args=get_value_at_index(wanvideotorchcompilesettings_82, 0),
                block_swap_args=get_value_at_index(wanvideoblockswap_238, 0),
                lora=get_value_at_index(wanvideoloraselectmulti_76, 0),
            )

            wanvideoclipvisionencode_87 = wanvideoclipvisionencode.process(
                strength_1=parse_arg(args.strength_1119),
                strength_2=parse_arg(args.strength_2120),
                crop=parse_arg(args.crop121),
                combine_embeds=parse_arg(args.combine_embeds122),
                force_offload=parse_arg(args.force_offload123),
                tiles=parse_arg(args.tiles124),
                ratio=parse_arg(args.ratio125),
                clip_vision=get_value_at_index(clipvisionloader_80, 0),
                image_1=get_value_at_index(imageresizekjv2_63, 0),
            )

            mathexpressionpysssss_144 = mathexpressionpysssss.evaluate(
                expression=parse_arg(args.expression126),
                a=get_value_at_index(mathexpressionpysssss_21, 0),
                b=get_value_at_index(intconstant_161, 0),
                prompt=PROMPT_DATA,
            )

            wanvideoanimateembeds_88 = wanvideoanimateembeds.process(
                width=get_value_at_index(imageresizekjv2_33, 1),
                height=get_value_at_index(imageresizekjv2_33, 2),
                num_frames=get_value_at_index(vhs_loadvideo_13, 1),
                force_offload=parse_arg(args.force_offload127),
                frame_window_size=get_value_at_index(mathexpressionpysssss_144, 0),
                colormatch=parse_arg(args.colormatch128),
                pose_strength=parse_arg(args.pose_strength129),
                face_strength=parse_arg(args.face_strength130),
                tiled_vae=parse_arg(args.tiled_vae131),
                vae=get_value_at_index(wanvideovaeloader_81, 0),
                clip_embeds=get_value_at_index(wanvideoclipvisionencode_87, 0),
                ref_images=get_value_at_index(imageresizekjv2_63, 0),
                pose_images=get_value_at_index(dwpreprocessor_53, 0),
                face_images=get_value_at_index(imagecropbymaskandresize_56, 0),
                bg_images=get_value_at_index(drawmaskonimage_43, 0),
                mask=get_value_at_index(blockifymask_39, 0),
            )

            wanvideotextembedbridge_125 = wanvideotextembedbridge.process(
                positive=get_value_at_index(cliptextencode_124, 0),
                negative=get_value_at_index(cliptextencode_122, 0),
            )

            wanvideosampler_90 = wanvideosampler.process(
                steps=parse_arg(args.steps132),
                cfg=parse_arg(args.cfg133),
                shift=parse_arg(args.shift134),
                seed=parse_arg(args.seed135),
                force_offload=parse_arg(args.force_offload136),
                scheduler=parse_arg(args.scheduler137),
                riflex_freq_index=parse_arg(args.riflex_freq_index138),
                denoise_strength=parse_arg(args.denoise_strength139),
                batched_cfg=parse_arg(args.batched_cfg140),
                rope_function=parse_arg(args.rope_function141),
                start_step=parse_arg(args.start_step142),
                end_step=parse_arg(args.end_step143),
                add_noise_to_samples=parse_arg(args.add_noise_to_samples144),
                model=get_value_at_index(wanvideomodelloader_75, 0),
                image_embeds=get_value_at_index(wanvideoanimateembeds_88, 0),
                text_embeds=get_value_at_index(wanvideotextembedbridge_125, 0),
            )

            wanvideodecode_91 = wanvideodecode.decode(
                enable_vae_tiling=parse_arg(args.enable_vae_tiling145),
                tile_x=parse_arg(args.tile_x146),
                tile_y=parse_arg(args.tile_y147),
                tile_stride_x=parse_arg(args.tile_stride_x148),
                tile_stride_y=parse_arg(args.tile_stride_y149),
                normalization=parse_arg(args.normalization150),
                vae=get_value_at_index(wanvideovaeloader_81, 0),
                samples=get_value_at_index(wanvideosampler_90, 0),
            )

            getimagesizeandcount_92 = getimagesizeandcount.getsize(
                image=get_value_at_index(wanvideodecode_91, 0)
            )

            vhs_videocombine_106 = vhs_videocombine.combine_video(
                frame_rate=get_value_at_index(primitivefloat_30, 0),
                loop_count=parse_arg(args.loop_count151),
                filename_prefix=parse_arg(args.filename_prefix152),
                format=parse_arg(args.format153),
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=parse_arg(args.pingpong154),
                save_output=parse_arg(args.save_output155),
                images=get_value_at_index(getimagesizeandcount_92, 0),
                audio=get_value_at_index(vhs_loadvideo_13, 2),
            )

            vhs_videocombine_141 = vhs_videocombine.combine_video(
                frame_rate=parse_arg(args.frame_rate156),
                loop_count=parse_arg(args.loop_count157),
                filename_prefix=parse_arg(args.filename_prefix158),
                format=parse_arg(args.format159),
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=False,
                trim_to_audio=False,
                pingpong=parse_arg(args.pingpong160),
                save_output=parse_arg(args.save_output161),
                images=get_value_at_index(imagecropbymaskandresize_56, 0),
            )

            cm_floattoint_176 = cm_floattoint.op(
                a=get_value_at_index(primitivefloat_30, 0)
            )

            rife_vfi_171 = rife_vfi.vfi(
                ckpt_name=parse_arg(args.ckpt_name162),
                clear_cache_after_n_frames=get_value_at_index(cm_floattoint_176, 0),
                multiplier=get_value_at_index(intconstant_29, 0),
                fast_mode=parse_arg(args.fast_mode163),
                ensemble=parse_arg(args.ensemble164),
                scale_factor=parse_arg(args.scale_factor165),
                frames=get_value_at_index(getimagesizeandcount_92, 0),
            )

            mathexpressionpysssss_189 = mathexpressionpysssss.evaluate(
                expression=parse_arg(args.expression166),
                a=get_value_at_index(primitivefloat_30, 0),
                b=get_value_at_index(intconstant_29, 0),
                prompt=PROMPT_DATA,
            )

            vhs_videocombine_178 = vhs_videocombine.combine_video(
                frame_rate=get_value_at_index(mathexpressionpysssss_189, 1),
                loop_count=parse_arg(args.loop_count167),
                filename_prefix=parse_arg(args.filename_prefix168),
                format=parse_arg(args.format169),
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=parse_arg(args.pingpong170),
                save_output=parse_arg(args.save_output171),
                images=get_value_at_index(rife_vfi_171, 0),
                audio=get_value_at_index(vhs_loadvideo_13, 2),
            )


if __name__ == "__main__":
    main()

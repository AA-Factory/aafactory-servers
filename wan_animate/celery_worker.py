import os
import sys
from celery import Celery
from celery.utils.log import get_task_logger
from wan_animate.wan_animate.cli_arguments.builder import ArgumentBuilder
from wan_animate.wan_animate.cli_arguments.config_animate import ArgumentConfigAnimate
from wan_animate.wan_animate.cli_arguments.config_replace import ArgumentConfigReplace
from wan_animate.worker_utils import (
    b64_to_bytes,
    bytes_to_b64,
    delete_files_from_folder,
    detect_file_extension,
    run_python_command,
    write_bytes_to_path,
)

logger = get_task_logger(__name__)
logger.setLevel("INFO")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

app = Celery(
    "wan_2_2_animate_worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)

GENERATE_ANIMATE_SCRIPT = "/app/wan_animate/workflows/workflow_animate.py"
GENERATE_REPLACE_SCRIPT = "/app/wan_animate/workflows/workflow_replace.py"
# Root directory for workflow (inside the docker container)
WORKFLOW_ROOT_DIR = "/app/wan_animate"
INPUT_PATH = "/app/wan_animate/ComfyUI/input/"
OUTPUT_PATH = "/app/wan_animate/ComfyUI/output/"
# Define name of the input files to be stored in Comfyui
INPUT_IMAGE_FILE_NAME = "image"
INPUT_VIDEO_FILE_NAME = "video"
# Define which videos we want to read after generation
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_PATH, "Wanimate_Interpolated_00001-audio.mp4")


@app.task(name="image_and_video_to_video_animate", queue="wan_animate")
def image_and_video_to_video_animate(
    image_bytes: str, video_bytes: str, user_args: dict = None
) -> dict:
    return image_and_video_to_video("animate", image_bytes, video_bytes, user_args)


@app.task(name="image_and_video_to_video_replace", queue="wan_replace")
def image_and_video_to_video_replace(
    image_bytes: str, video_bytes: str, user_args: dict = None
) -> dict:
    return image_and_video_to_video("replace", image_bytes, video_bytes, user_args)


def image_and_video_to_video(
    generation_type: str, image_bytes: str, video_bytes: str, user_args: dict = None
) -> dict:
    """
    Accepts image and video as base64-encoded strings.
    Returns dictionary with a base64-encoded strings containing the generated output (MP4).
    """
    image_bytes = b64_to_bytes(image_bytes)
    video_bytes = b64_to_bytes(video_bytes)
    output_video_bytes = _run_pipeline(
        generation_type, image_bytes, video_bytes, user_args or {}
    )
    return bytes_to_b64(output_video_bytes)


def _load_and_save_inputs(image_bytes: bytes, video_bytes: bytes) -> None:
    """Determine extensions, build paths, and write files"""

    image_ext = detect_file_extension(image_bytes)
    video_ext = detect_file_extension(video_bytes)

    image_path = os.path.join(INPUT_PATH, f"{INPUT_IMAGE_FILE_NAME}.{image_ext}")
    video_path = os.path.join(INPUT_PATH, f"{INPUT_VIDEO_FILE_NAME}.{video_ext}")

    write_bytes_to_path(image_bytes, path=image_path)
    write_bytes_to_path(video_bytes, path=video_path)
    return image_path, video_path


def _run_pipeline(
    generation_type: str, image_bytes: bytes, video_bytes: bytes, user_args: dict
) -> bytes:
    """
    Core pipeline: cleanup, write incoming bytes to files, run generate, read output bytes.
    Always returns bytes of the created output file.
    """

    try:
        _cleanup_old_inputs_outputs()
        # 1) Write inputs to input folder
        image_path, video_path = _load_and_save_inputs(image_bytes, video_bytes)
        logger.info(f"Wrote input bytes to {image_path} and {video_path}")
        logger.info(f"Will write generated output to: {OUTPUT_PATH}")

        # 2) Set up environment and run workflow
        if generation_type == "replace":
            generate_script = GENERATE_REPLACE_SCRIPT
            argument_config = ArgumentConfigAnimate()
        else:
            generate_script = GENERATE_ANIMATE_SCRIPT
            argument_config = ArgumentConfigReplace()

        argument_builder = ArgumentBuilder(argument_config)
        system_cli_args = argument_builder.build_system_cli_args(image_path, video_path)
        user_cli_args = argument_builder.build_user_cli_args(user_args)
        generate_command = (
            [sys.executable, generate_script] + system_cli_args + user_cli_args
        )

        # 3) Start generation
        logger.info("Generation started...")
        run_python_command(generate_command, cwd=WORKFLOW_ROOT_DIR)
        logger.info("Generation finished.")

        # 4) Read output file into bytes and return
        return _fetch_output()

    except Exception as e:
        logger.exception("Pipeline failed")
        raise RuntimeError(f"Pipeline failed: {e}")


def _fetch_output() -> bytes:
    """Reads the generated output video file and returns its bytes."""

    logger.info(f"Output folder has following files:")
    for root, _, files in os.walk(OUTPUT_PATH):
        for file in files:
            logger.info(os.path.join(root, file))
    with open(OUTPUT_VIDEO_PATH, "rb") as f:
        return f.read()


def _cleanup_old_inputs_outputs():
    """Cleans up old input and output files to avoid interference."""

    delete_files_from_folder(INPUT_PATH)
    delete_files_from_folder(OUTPUT_PATH)

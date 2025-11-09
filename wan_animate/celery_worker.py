import os
import sys
from celery import Celery
from celery.utils.log import get_task_logger
from wan_animate.arg_builder import build_system_cli_args, build_user_cli_args
from wan_animate.worker_utils import (
    b64_to_bytes,
    bytes_to_b64,
    delete_files_from_folder,
    detect_file_extension,
    run_python_command,
    write_bytes_to_path,
)
from wan_animate.workflow import main

logger = get_task_logger(__name__)
logger.setLevel("INFO")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

app = Celery(
    "wan_2_2_animate_worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)
# Use JSON serializer and encode binary inputs/outputs with base64 so messages are safe and JSON-serializable.
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],  # only accept JSON-serialized tasks/results
)
app.conf.task_queues = {
    "animate": {"exchange": "animate", "routing_key": "animate"},
}
app.conf.broker_transport_options = {
    "socket_keepalive": True,
    "socket_timeout": 30,  # seconds for socket ops
    "socket_connect_timeout": 10,
}
app.conf.result_backend_transport_options = {
    "socket_keepalive": True,
    "socket_timeout": 30,
}

GENERATE_SCRIPT = "/app/wan_animate/workflow.py"
# Root directory for workflow (inside the docker container)
WORKFLOW_ROOT_DIR = "/app/wan_animate"
INPUT_PATH = "/app/wan_animate/ComfyUI/input/"
OUTPUT_PATH = "/app/wan_animate/ComfyUI/output/"
# Define paths for input files
INPUT_IMAGE_FILE_NAME = "image"
INPUT_VIDEO_FILE_NAME = "video"
# Define which videos we want to read after generation
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_PATH, "Wanimate_Interpolated_00001-audio.mp4")


@app.task(name="image_and_video_to_video", queue="wan_animate")
def image_and_video_to_video(
    image_bytes: str, video_bytes: str, user_args: dict = None
) -> dict:
    """
    Accepts image and video as base64-encoded strings.
    Returns dictionary with a base64-encoded strings containing the generated output (MP4).
    """
    image_bytes = b64_to_bytes(image_bytes)
    video_bytes = b64_to_bytes(video_bytes)
    output_video_bytes = _run_pipeline(image_bytes, video_bytes, user_args or {})
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


def _run_pipeline(image_bytes: bytes, video_bytes: bytes, user_args: dict) -> bytes:
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
        system_cli_args = build_system_cli_args(image_path, video_path)
        user_cli_args = build_user_cli_args(user_args)
        # generate_command = (
        #     [sys.executable, GENERATE_SCRIPT] + system_cli_args + user_cli_args
        # )

        # 3) Start generation
        logger.info("Generation started...")
        main(**{**system_cli_args, **user_cli_args})
        # run_python_command(generate_command, cwd=WORKFLOW_ROOT_DIR)
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

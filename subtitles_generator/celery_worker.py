import os
from celery import Celery
from celery.utils.log import get_task_logger

from subtitles_generator.generate_subtitles_pipeline import generate_subtitles_pipeline
from subtitles_generator.worker_utils import (
    b64_to_bytes,
    bytes_to_b64,
    delete_files_from_folder,
    detect_file_extension,
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

DATA_DIR_PATH = "/app/subtitle_generator/data/"
os.makedirs(DATA_DIR_PATH, exist_ok=True)

@app.task(name="generate_subtitles", queue="generate_subtitles")
def generate_subtitles(video_bytes: str, user_args: dict = None) -> dict:
    """
    Accepts video as base64-encoded strings.
    Returns <INSERT RETURN TYPE>.
    """
    video_bytes = b64_to_bytes(video_bytes)
    output_video_bytes = _run_pipeline(video_bytes, user_args or {})
    return bytes_to_b64(output_video_bytes)


def _run_pipeline(video_bytes: bytes, user_args: dict) -> bytes:
    """
    Core pipeline: cleanup, write incoming bytes to files, run generate, read output bytes.
    Always returns bytes of the created output file.
    """

    try:
        _cleanup_old_inputs_outputs()
        # 1) Write inputs to input folder
        video_path = _load_and_save_inputs(video_bytes)
        logger.info(f"Wrote input bytes to {video_path}")
        logger.info(f"Will write generated output to: {DATA_DIR_PATH}")

        # 2) Run subtitle generation script
        srt_path = generate_subtitles_pipeline(video_path, user_args)

        # 3) Read output file into bytes and return
        return _fetch_output(srt_path)

    except Exception as e:
        logger.exception("Pipeline failed")
        raise RuntimeError(f"Pipeline failed: {e}")


def _load_and_save_inputs(video_bytes: bytes) -> None:
    """Determine extensions, build paths, and write files"""

    video_ext = detect_file_extension(video_bytes)
    video_path = os.path.join(DATA_DIR_PATH, f"video.{video_ext}")

    write_bytes_to_path(video_bytes, path=video_path)
    return video_path


def _fetch_output(srt_path: str) -> bytes:
    """Reads the generated output file and returns its bytes."""

    logger.info(f"Output folder has following files:")
    for root, _, files in os.walk(DATA_DIR_PATH):
        for file in files:
            logger.info(os.path.join(root, file))
    with open(srt_path, "rb") as f:
        return f.read()


def _cleanup_old_inputs_outputs():
    """Cleans up old input and output files to avoid interference."""

    delete_files_from_folder(DATA_DIR_PATH)

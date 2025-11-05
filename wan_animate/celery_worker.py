import os
import sys
import shutil
import subprocess
import base64
from celery import Celery
from celery.utils.log import get_task_logger

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

GENERATE_SCRIPT = "/app/comfyui_logic/workflow.py"
GENERATE_SCRIPT = os.path.abspath(GENERATE_SCRIPT)
if not os.path.exists(GENERATE_SCRIPT):
    raise RuntimeError(f"Generate script not found: {GENERATE_SCRIPT}")
# try to make the script executable (harmless if already executable or failing on restrictive FS)
if not os.access(GENERATE_SCRIPT, os.X_OK):
    try:
        os.chmod(GENERATE_SCRIPT, 0o755)
    except Exception:
        logger.warning(
            f"Could not chmod {GENERATE_SCRIPT}; ensure it is executable if needed."
        )
# Root directory for workflow (inside the docker container)
WORKFLOW_ROOT_DIR = "/app/comfyui_logic"
INPUT_PATH = "/app/comfyui_logic/ComfyUI/input/"
OUTPUT_PATH = "/app/comfyui_logic/ComfyUI/output/"
# Define paths for input files
IMAGE_PATH = os.path.join(INPUT_PATH, "image.jpeg")
VIDEO_PATH = os.path.join(INPUT_PATH, "video.mp4")
# Define which videos we want to read after generation
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_PATH, "Wanimate_Interpolated_00001-audio.mp4")


@app.task(name="tasks.animate", queue="animate")
def animate(image_bytes: str, video_bytes: str) -> dict:
    """
    Accepts image and video as base64-encoded strings.
    Returns dictionary with a base64-encoded strings containing the generated output (MP4).
    """
    image_bytes = _b64_to_bytes(image_bytes)
    video_bytes = _b64_to_bytes(video_bytes)
    output_video_bytes = _run_pipeline(image_bytes, video_bytes)
    return _bytes_to_b64(output_video_bytes)


def _b64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data)


def _bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _run_pipeline(image_bytes: bytes, video_bytes: bytes) -> bytes:
    """
    Core pipeline: cleanup, write incoming bytes to files, run generate, read output bytes.
    Always returns bytes of the created output file.
    """

    try:
        _cleanup_old_inputs_outputs()
        # 1) Write inputs to input folder
        _write_bytes_to_path(image_bytes, path=IMAGE_PATH)
        _write_bytes_to_path(video_bytes, path=VIDEO_PATH)
        logger.info(f"Wrote input bytes to {IMAGE_PATH} and {VIDEO_PATH}")
        logger.info(f"Will write generated output to: {OUTPUT_PATH}")

        # 2) Set up environment and run workflow
        generate_command = [sys.executable, GENERATE_SCRIPT]
        _run_wan_command(generate_command, cwd=WORKFLOW_ROOT_DIR)
        logger.info("Generation finished.")

        # 3) Read output file into bytes and return
        logger.info(f"Output folder has following files:")
        for root, _, files in os.walk(OUTPUT_PATH):
            for file in files:
                logger.info(os.path.join(root, file))
        with open(OUTPUT_VIDEO_PATH, "rb") as f:
            output_video_bytes = f.read()

        return output_video_bytes

    except Exception as e:
        logger.exception("Pipeline fail ed")
        raise RuntimeError(f"Pipeline failed: {e}")


def _cleanup_old_inputs_outputs():
    """Cleans up old input and output files to avoid interference."""

    for folder in [INPUT_PATH, OUTPUT_PATH]:
        # Remove every entry inside the folder but keep the folder itself.
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                # If it's a file or symlink, remove it; if it's a directory, rmtree it.
                if os.path.islink(path) or os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    # Unknown file type: try removing as file.
                    os.remove(path)
                logger.info(f"Removed: {path}")
            except Exception:
                logger.warning(f"Could not remove: {path}", exc_info=True)


def _write_bytes_to_path(data: bytes, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

def _run_wan_command(command_args: list, cwd: str, env: dict = None) -> str:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    logger.info(f"Executing command: {' '.join(command_args)} in CWD: {cwd}")
    try:
        res = subprocess.run(
            command_args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            env=full_env,
        )
        if res.stdout:
            logger.debug(res.stdout)
        if res.stderr:
            logger.warning(res.stderr)
        return res.stdout
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Command failed: returncode={e.returncode}. stdout={e.stdout} stderr={e.stderr}"
        )
        raise RuntimeError(f"Wan2.2 command failed: {e.stderr or e.stdout}")
    except FileNotFoundError:
        logger.error("Executable or script not found.")
        raise RuntimeError("Command executable or script not found.")
    except Exception as e:
        logger.exception("Unexpected error running command")
        raise RuntimeError(f"Unexpected error: {e}")
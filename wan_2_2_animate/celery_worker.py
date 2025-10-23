import os
import sys
import uuid
import shutil
import subprocess
import tempfile
import base64
from typing import Tuple, Union
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
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],  # only accept JSON-serialized tasks/results
)
app.conf.task_queues = {
    'animate': {'exchange': 'animate', 'routing_key': 'animate'},
}

TASK_TYPE = "animate-14B"
# path to the generation script (inside the docker container)
GENERATE_SCRIPT = "/app/comfyui_logic/workflow.py"
GENERATE_SCRIPT = os.path.abspath(GENERATE_SCRIPT)
if not os.path.exists(GENERATE_SCRIPT):
    raise RuntimeError(f"Generate script not found: {GENERATE_SCRIPT}")
# try to make the script executable (harmless if already executable or failing on restrictive FS)
if not os.access(GENERATE_SCRIPT, os.X_OK):
    try:
        os.chmod(GENERATE_SCRIPT, 0o755)
    except Exception:
        logger.warning(f"Could not chmod {GENERATE_SCRIPT}; ensure it is executable if needed.")
WORKFLOW_ROOT_DIR = "/app/comfyui_logic"

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
        logger.error(f"Command failed: returncode={e.returncode}. stdout={e.stdout} stderr={e.stderr}")
        raise RuntimeError(f"Wan2.2 command failed: {e.stderr or e.stdout}")
    except FileNotFoundError:
        logger.error("Executable or script not found.")
        raise RuntimeError("Command executable or script not found.")
    except Exception as e:
        logger.exception("Unexpected error running command")
        raise RuntimeError(f"Unexpected error: {e}")


def _guess_ext_from_bytes(data: bytes, default: str = ".bin") -> str:
    if not data or len(data) < 12:
        return default
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return ".png"
    if data.startswith(b'\xff\xd8'):
        return ".jpg"
    if data.startswith(b'GIF8'):
        return ".gif"
    if data[0:4] == b'RIFF' and b'WEBP' in data[8:16]:
        return ".webp"
    if b'ftyp' in data[4:12]:
        return ".mp4"
    if data.startswith(b'\x1A\x45\xDF\xA3'):
        return ".mkv"
    if data[0:4] == b'RIFF' and data[8:12] == b'AVI ':
        return ".avi"
    return default


def _write_bytes_to_path(data: bytes, prefix: str, suggested_ext: str = None) -> str:
    """
    Write bytes directly to a target path if `prefix` contains a directory component (i.e. looks like a path).
    Otherwise fall back to creating a NamedTemporaryFile (preserves original behaviour).
    Returns the path to the written file.
    """
    # If prefix looks like a path (has a directory), use it as the target path.
    target_dir = os.path.dirname(prefix)
    if target_dir:
        target = prefix
        base, ext = os.path.splitext(target)
        if not ext and suggested_ext:
            target = base + suggested_ext
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return target


def _to_bytes(data: Union[str, bytes], name: str = "data") -> bytes:
    """
    Ensure the given input is bytes.
    - If it's bytes, return as-is (helps in-process/backwards compatibility).
    - If it's str, treat it as base64 and decode to bytes.
    Raises:
        TypeError if input is neither bytes nor str.
        ValueError if base64 decoding fails.
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception as e:
            logger.error(f"Failed to base64-decode {name}: {e}")
            raise ValueError(f"Invalid base64 for {name}: {e}")
    raise TypeError(f"{name} must be bytes or base64-encoded string")


def _bytes_to_b64(data: bytes) -> str:
    """Encode bytes to base64 string for JSON-safe transport."""
    return base64.b64encode(data).decode("ascii")


@app.task(name="tasks.animate", queue="animate")
def animate(source_image_b64_or_bytes: Union[str, bytes], driving_video_b64_or_bytes: Union[str, bytes], seed: int = 42) -> str:
    """
    Accepts source image and driving video as base64-encoded strings (recommended).
    For backward/in-process compatibility, raw bytes are also accepted.
    Returns a base64-encoded string containing the generated output (MP4).
    """
    source_bytes = _to_bytes(source_image_b64_or_bytes, "source_image")
    driving_bytes = _to_bytes(driving_video_b64_or_bytes, "driving_video")
    out_bytes = _bytes_pipeline(source_bytes, driving_bytes, seed, replace_flag=False)
    return _bytes_to_b64(out_bytes)


def _bytes_pipeline(source_bytes: bytes, driving_bytes: bytes, seed: int, replace_flag: bool) -> bytes:
    """
    Core pipeline: write incoming bytes to temp files, run preprocess + generate, read output bytes, cleanup.
    Always returns bytes of the created output file.
    """
    temp_paths = []
    output_path = None

    try:
        # 1) Write inputs to temp files
        src_path = _write_bytes_to_path(source_bytes, prefix="/app/comfyui_logic/ComfyUI/input/image.jpeg", suggested_ext=".jpeg")
        temp_paths.append(src_path)
        vid_path = _write_bytes_to_path(driving_bytes, prefix="/app/comfyui_logic/ComfyUI/input/video.mp4", suggested_ext=".mp4")
        temp_paths.append(vid_path)
        logger.info(f"Wrote input bytes to {src_path} and {vid_path}")

        output_path = "/app/comfyui_logic/ComfyUI/output/"
        output_file1 = os.path.join(output_path, "Wanimate_00001-audio.mp4")
        output_file2 = os.path.join(output_path, "Wanimate_Interpolated_00001.mp4")
        
        temp_paths.append(output_file1)  # treat output as temp to remove later (after reading)
        temp_paths.append(output_file2)  # treat output as temp to remove later (after reading)
        logger.info(f"Will write generated output to: {output_path}")

        # 4) Generation command
        generate_command = [
            sys.executable,
            GENERATE_SCRIPT
        ]

        _run_wan_command(generate_command, cwd=WORKFLOW_ROOT_DIR)
        logger.info("Generation finished.")

        # 5) Read output file into bytes and return
        with open(output_file1, "rb") as f:
            out_bytes1 = f.read()
        with open(output_file2, "rb") as f:
            out_bytes2 = f.read()

        return out_bytes1

    except Exception as e:
        logger.exception("Pipeline fail ed")
        raise

    finally:
        # Cleanup all created files and directories
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    logger.info(f"Removed temp file: {p}")
            except Exception:
                logger.warning(f"Could not remove temp file: {p}", exc_info=True)
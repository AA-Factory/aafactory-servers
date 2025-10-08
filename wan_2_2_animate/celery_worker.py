import os
import sys
import uuid
import shutil
import subprocess
import tempfile
import base64
from typing import Tuple, Union
import torch
from celery import Celery
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)
logger.setLevel("INFO")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    'replace': {'exchange': 'replace', 'routing_key': 'replace'},
}

WAN_ROOT_DIR = "/app/Wan2.2"
MODEL_PATH = "/.weights/Wan2.2-Animate-14B"
PREPROCESS_SCRIPT = os.path.join(WAN_ROOT_DIR, 'wan/modules/animate/preprocess/preprocess_data.py')
GENERATE_SCRIPT = os.path.join(WAN_ROOT_DIR, 'generate.py')
TASK_TYPE = "animate-14B"


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


def _write_bytes_to_tempfile(data: bytes, prefix: str, suggested_ext: str = None) -> str:
    ext = suggested_ext or _guess_ext_from_bytes(data, default=".bin")
    with tempfile.NamedTemporaryFile(prefix=f"wan_{prefix}_", suffix=ext, delete=False) as tf:
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
        return tf.name


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


@app.task(name="tasks.replace", queue="replace")
def replace(source_image_b64_or_bytes: Union[str, bytes], driving_video_b64_or_bytes: Union[str, bytes], seed: int = 42) -> str:
    """
    Accepts source image and driving video as base64-encoded strings (recommended).
    For backward/in-process compatibility, raw bytes are also accepted.
    Returns a base64-encoded string containing the generated output (MP4).
    """
    source_bytes = _to_bytes(source_image_b64_or_bytes, "source_image")
    driving_bytes = _to_bytes(driving_video_b64_or_bytes, "driving_video")
    out_bytes = _bytes_pipeline(source_bytes, driving_bytes, seed, replace_flag=True)
    return _bytes_to_b64(out_bytes)


def _bytes_pipeline(source_bytes: bytes, driving_bytes: bytes, seed: int, replace_flag: bool) -> bytes:
    """
    Core pipeline: write incoming bytes to temp files, run preprocess + generate, read output bytes, cleanup.
    Always returns bytes of the created output file.
    """
    temp_paths = []
    output_path = None
    preprocess_dir = None

    try:
        # 1) Write inputs to temp files
        src_path = _write_bytes_to_tempfile(source_bytes, prefix="source", suggested_ext=".png")
        temp_paths.append(src_path)
        vid_path = _write_bytes_to_tempfile(driving_bytes, prefix="driving", suggested_ext=".mp4")
        temp_paths.append(vid_path)
        logger.info(f"Wrote input bytes to {src_path} and {vid_path}")

        # 2) Preprocess into temporary directory
        preprocess_dir = tempfile.mkdtemp(prefix="wan_preprocess_")
        logger.info(f"Created preprocess dir: {preprocess_dir}")

        preprocess_command = [
            sys.executable,
            PREPROCESS_SCRIPT,
            "--ckpt_path", os.path.join(MODEL_PATH, 'process_checkpoint'),
            "--video_path", vid_path,
            "--refer_path", src_path,
            "--save_path", preprocess_dir,
            "--resolution_area", "854", "480",
            "--fps", "15",
        ]

        if replace_flag:
            preprocess_command.extend([
                "--iterations", "1",
                "--k", "3",
                "--w_len", "1",
                "--h_len", "1",
                "--replace_flag",
            ])
        else:
            preprocess_command.extend(["--retarget_flag"])

        _run_wan_command(preprocess_command, cwd=WAN_ROOT_DIR)
        logger.info("Preprocessing finished.")

        # 3) Prepare output tempfile (we will read and then delete it)
        out_tmp = tempfile.NamedTemporaryFile(prefix="wan_output_", suffix=".mp4", delete=False)
        output_path = out_tmp.name
        out_tmp.close()
        temp_paths.append(output_path)  # treat output as temp to remove later (after reading)
        logger.info(f"Will write generated output to: {output_path}")

        # 4) Generation command
        generate_command = [
            sys.executable,
            GENERATE_SCRIPT,
            "--task", TASK_TYPE,
            "--ckpt_dir", MODEL_PATH,
            "--src_root_path", preprocess_dir,
            "--refert_num", "1",
            "--save_file", output_path,
            "--offload_model", "True",
            "--t5_cpu",
            "--convert_model_dtype",
        ]

        if replace_flag:
            generate_command.extend(["--replace_flag", "--use_relighting_lora"])

        _run_wan_command(generate_command, cwd=WAN_ROOT_DIR)
        logger.info("Generation finished.")

        # 5) Read output file into bytes and return
        with open(output_path, "rb") as f:
            out_bytes = f.read()

        return out_bytes

    except Exception as e:
        logger.exception("Pipeline failed")
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

        if preprocess_dir and os.path.exists(preprocess_dir):
            try:
                shutil.rmtree(preprocess_dir)
                logger.info(f"Removed preprocess dir: {preprocess_dir}")
            except Exception:
                logger.warning(f"Could not remove preprocess dir: {preprocess_dir}", exc_info=True)

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache.")
            except Exception:
                logger.warning("Failed to clear CUDA cache.", exc_info=True)

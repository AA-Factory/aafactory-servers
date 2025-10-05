import os
import sys
import uuid
import shutil
import subprocess
import torch # Still needed for `torch.cuda.is_available()`
from celery import Celery
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Celery(
    "wan_2_2_animate_worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)
app.conf.task_queues = {
    'animate': {'exchange': 'animate', 'routing_key': 'animate'},
    'replace': {'exchange': 'replace', 'routing_key': 'replace'},
}

# --- Global Paths for Wan2.2 Scripts and Models ---
WAN_ROOT_DIR = "/app/Wan2.2"
MODEL_PATH = "/.weights/Wan2.2-Animate-14B" # This is passed to Wan2.2 scripts
PREPROCESS_SCRIPT = os.path.join(WAN_ROOT_DIR, 'wan/modules/animate/preprocess/preprocess_data.py')
GENERATE_SCRIPT = os.path.join(WAN_ROOT_DIR, 'generate.py')
TASK_TYPE = "animate-14B" # Defined by Wan2.2 documentation for this model

# --- Helper to run subprocess commands ---
def _run_wan_command(command_args: list, cwd: str, env: dict = None):
    """
    Executes a shell command for the Wan2.2 project.
    Raises an exception if the command fails.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env) # Merge provided env vars

    logger.info(f"Executing command: {' '.join(command_args)} in CWD: {cwd}")
    try:
        result = subprocess.run(
            command_args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True, # Raise CalledProcessError for non-zero exit codes
            env=full_env # Pass environment variables
        )
        logger.debug(f"Command stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr: {result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {' '.join(command_args)}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Wan2.2 command failed: {e.stderr or e.stdout}")
    except FileNotFoundError:
        logger.error(f"Command executable not found. Is 'python' in PATH or script at '{command_args[1]}' correct?")
        raise RuntimeError(f"Command executable or script not found: {command_args[0]} {command_args[1]}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during command execution: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during command execution: {e}")

# --- Celery Task Definitions ---

@app.task(name="tasks.animate", queue="animate")
def animate(source_image_path: str, driving_video_path: str, output_path: str, seed: int = 42):
    """
    Task to generate an animation video using subprocess calls to Wan2.2 scripts.
    """
    return _process_and_generate_via_commands(
        source_image_path=source_image_path,
        driving_video_path=driving_video_path,
        output_path=output_path,
        seed=seed,
        replace_flag=False
    )

@app.task(name="tasks.replace", queue="replace")
def replace(source_image_path: str, driving_video_path: str, output_path: str, seed: int = 42):
    """
    Task to generate a replacement video using subprocess calls to Wan2.2 scripts.
    """
    return _process_and_generate_via_commands(
        source_image_path=source_image_path,
        driving_video_path=driving_video_path,
        output_path=output_path,
        seed=seed,
        replace_flag=True
    )

def _process_and_generate_via_commands(source_image_path: str, driving_video_path: str, output_path: str, seed: int, replace_flag: bool):
    """
    Orchestrates preprocessing and generation using subprocess calls to Wan2.2's scripts.
    """
    preprocess_dir = f"/tmp/wan_preprocess_{uuid.uuid4()}"
    os.makedirs(preprocess_dir, exist_ok=True)
    
    try:
        # --- 1. Preprocessing Command ---
        logger.info(f"Starting preprocessing for {driving_video_path} using Wan2.2 script. Output: {preprocess_dir}")
        preprocess_command = [
            sys.executable,
            PREPROCESS_SCRIPT,
            "--ckpt_path", os.path.join(MODEL_PATH, 'process_checkpoint'),
            "--video_path", driving_video_path,
            "--refer_path", source_image_path,
            "--save_path", preprocess_dir,
            "--resolution_area", "1280", "720", # Note: resolution_area takes two args in CLI
        ]
        
        if replace_flag:
            preprocess_command.extend([
                "--iterations", "3",
                "--k", "7",
                "--w_len", "1",
                "--h_len", "1",
                "--replace_flag",
            ])
        else: # Animation mode
            preprocess_command.extend([
                "--retarget_flag",
                # "--use_flux",
            ])
            
        _run_wan_command(preprocess_command, cwd=WAN_ROOT_DIR) # Run from project root
        logger.info("Preprocessing via command completed successfully.")

        # --- 2. Generation Command ---
        logger.info(f"Starting generation for preprocessed data in {preprocess_dir} using Wan2.2 script.")
        generate_command = [
            sys.executable,
            GENERATE_SCRIPT,
            "--task", TASK_TYPE,
            "--ckpt_dir", MODEL_PATH,
            "--src_root_path", preprocess_dir,
            "--refert_num", "1",
            "--output_path", output_path, # Custom output path from your task
            "--seed", str(seed),
            # Add general efficiency flags as per documentation
            "--offload_model", "True",
            "--convert_model_dtype",
            "--t5_cpu", "True", # Add t5_cpu if you were using it previously to save memory
        ]

        if replace_flag:
            generate_command.extend([
                "--replace_flag",
                "--use_relighting_lora",
            ])
            
        _run_wan_command(generate_command, cwd=WAN_ROOT_DIR) # Run from project root
        logger.info(f"Generation via command successful. Video saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"An error occurred during the overall pipeline execution: {e}", exc_info=True)
        raise # Re-raise to ensure Celery task is marked as failed
    finally:
        # --- 3. Cleanup ---
        if os.path.exists(preprocess_dir):
            shutil.rmtree(preprocess_dir)
            logger.info(f"Cleaned up temporary directory: {preprocess_dir}")
        
        # Clearing CUDA cache can still be beneficial after subprocess calls if memory is tight.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache.")
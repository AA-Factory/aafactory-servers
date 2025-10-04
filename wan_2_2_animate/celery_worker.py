import os
import uuid
import shutil
import torch
from celery import Celery
from celery.utils.log import get_task_logger

# --- Direct Imports from the Cloned 'wan2.2' Repository ---
# Import the main animation pipeline
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
# Import the main function from the preprocessing script
from scripts.preprocess_data import main as preprocess_main

logger = get_task_logger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

DEVICE = "cuda"

app = Celery(
    "wan_2_2_animate_worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)
# Define the queues
app.conf.task_queues = {
    'animate': {'exchange': 'animate', 'routing_key': 'animate'},
    'replace': {'exchange': 'replace', 'routing_key': 'replace'},
}

# --- Global Model Loading (runs once per worker process) ---
MODEL_PATH = "/app/models/Wan2.2-Animate-14B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK_TYPE = "animate-14B"
WAN_ANIMATE_PIPELINE = None

logger.info(f"Attempting to load model '{TASK_TYPE}' from '{MODEL_PATH}' onto device '{DEVICE}'...")
try:
    # Instantiate the main pipeline from the wan library
    WAN_ANIMATE_PIPELINE = wan.WanAnimate(
        config=WAN_CONFIGS[TASK_TYPE],
        checkpoint_dir=MODEL_PATH,
        device_id=0 if DEVICE == "cuda" else None,
        # Set t5_cpu=True to save GPU memory if needed
        t5_cpu=True
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not load WanAnimate model. Worker will not be able to process tasks. Error: {e}", exc_info=True)

# --- Celery Task Definitions ---

@app.task(name="tasks.animate", queue="animate")
def animate(source_image_path: str, driving_video_path: str, output_path: str, seed: int = 42):
    """
    Task to generate an animation video. It preprocesses the data and then
    runs the animation pipeline with replace_flag=False.
    """
    return _run_generation_pipeline(
        source_image_path=source_image_path,
        driving_video_path=driving_video_path,
        output_path=output_path,
        seed=seed,
        replace_flag=False  # Key difference for the 'animate' task
    )

@app.task(name="tasks.replace", queue="replace")
def replace(source_image_path: str, driving_video_path: str, output_path: str, seed: int = 42):
    """
    Task to generate a replacement video. It preprocesses the data and then
    runs the animation pipeline with replace_flag=True.
    """
    return _run_generation_pipeline(
        source_image_path=source_image_path,
        driving_video_path=driving_video_path,
        output_path=output_path,
        seed=seed,
        replace_flag=True  # Key difference for the 'replace' task
    )

def _run_generation_pipeline(source_image_path: str, driving_video_path: str, output_path: str, seed: int, replace_flag: bool):
    """
    Internal function that encapsulates the shared logic for both animate and replace tasks.
    """
    if not WAN_ANIMATE_PIPELINE:
        error_msg = "Model is not loaded. Cannot process task."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 1. Preprocessing Step
    # Create a unique temporary directory for the preprocessed data
    preprocess_dir = f"/tmp/preprocess_{uuid.uuid4()}"
    os.makedirs(preprocess_dir, exist_ok=True)
    logger.info(f"Starting preprocessing. Output directory: {preprocess_dir}")

    try:
        # Create a namespace object to mimic argparse for the preprocess script
        class PreprocessArgs:
            src_path = driving_video_path
            ref_path = source_image_path
            save_path = preprocess_dir
            img_size = 512
            mode = 'crop'

        # Call the imported preprocessing function directly
        preprocess_main(PreprocessArgs())
        logger.info("Preprocessing completed successfully.")

        # 2. Generation Step
        logger.info(f"Starting generation with replace_flag={replace_flag}...")
        video_tensor = WAN_ANIMATE_PIPELINE.generate(
            src_root_path=preprocess_dir,
            replace_flag=replace_flag,
            seed=seed,
            sampling_steps=50,      # Default from docs
            guide_scale=7.5,        # Default from docs
            offload_model=True      # Recommended for memory saving
        )

        # 3. Save Output
        save_video(
            tensor=video_tensor[None],
            save_file=output_path,
            fps=WAN_CONFIGS[TASK_TYPE].sample_fps,
            normalize=True,
            value_range=(-1, 1)
        )
        logger.info(f"Generation successful. Video saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"An error occurred during the generation pipeline: {e}", exc_info=True)
        # Re-raise the exception so the task fails and can be retried/handled
        raise e
    finally:
        # 4. Cleanup
        # Always remove the temporary preprocessing directory
        if os.path.exists(preprocess_dir):
            shutil.rmtree(preprocess_dir)
            logger.info(f"Cleaned up temporary directory: {preprocess_dir}")
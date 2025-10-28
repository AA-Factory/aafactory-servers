import base64
import time
import warnings
import logging

import torch

from kandinsky.kandinsky import get_T2V_pipeline
from pathlib import Path


MODULE_PATH = Path(__file__).parent
CONFIGS_PATH = MODULE_PATH.parent / "kandinsky/configs/config_5s_sft.yaml"
VIDEO_DURATION = 5  # seconds
SAVE_FILE_PATH = "test.mp4"
ASPECT_RATIOS = {
    "1:1": (512, 512),
    "2:3": (512, 768),
    "3:2": (768, 512),
}

def run_text_to_video(prompt: str, video_aspect_ratio: str) -> bytes:
    """
    Generate a video from a text prompt using the configured text-to-video pipeline.

    This function prepares and runs a text-to-video (T2V) pipeline with preset
    configuration options (GPU device mapping, offload/magcache/quantization flags,
    and attention engine). It measures and logs the generation time and saves the
    resulting video to the module-level SAVE_FILE_PATH.

        prompt (str): A natural language prompt describing the content of the
            desired video.

    Behavior:
        - Disables library warnings before pipeline initialization.
        - Initializes or retrieves the T2V pipeline with the configured options.
        - Invokes the pipeline with global settings (VIDEO_DURATION, args.width,
          args.height, scheduler_scale, expand_prompts, etc.).
        - Saves the generated video to SAVE_FILE_PATH and logs elapsed time and
          the save location.

        bytes: Binary content of the generated video file. The video file is also
            written to SAVE_FILE_PATH. If generation fails, an exception is raised.

    Raises:
        RuntimeError: If the pipeline cannot be initialized or the generation step
            fails (propagates underlying exceptions from the pipeline).

    Run text-to-speech processing.

    Args:
        voice_sample (str): A sample of the voice to mimic.
        text (str): The text to convert to speech.
        language (dict): Language settings for the TTS.

    Returns:
        str: a base64-encoded string.
    """
    _disable_warnings()

    pipe = get_T2V_pipeline(
        device_map={"dit": "cuda:0", "vae": "cuda:0",
                    "text_embedder": "cuda:0"},
        conf_path=CONFIGS_PATH,
        offload=True,
        magcache=True,
        quantized_qwen=False,
        attention_engine="auto",
    )
    _ = pipe(prompt,
             time_length=VIDEO_DURATION,
             width=ASPECT_RATIOS[video_aspect_ratio][0],
             height=ASPECT_RATIOS[video_aspect_ratio][1],
             num_steps=None,
             guidance_weight=None,
             scheduler_scale=5.0,
             expand_prompts=1,
             save_path=SAVE_FILE_PATH)
    with open(SAVE_FILE_PATH, "rb") as f:
        video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes)
    return video_base64


def _disable_warnings():
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    torch._logging.set_logs(
        dynamo=logging.ERROR,
        dynamic=logging.ERROR,
        aot=logging.ERROR,
        inductor=logging.ERROR,
        guards=False,
        recompiles=False
    )

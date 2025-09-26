import random
from io import BytesIO
from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
import torch
import base64
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model


MODEL_NAME = "Qwen/Qwen-Image"
CPU_OFFLOAD = True
CPU_OFFLOAD_BLOCKS = 16
PIN_MEMORY = False

IMAGE_QUALITY_TO_STEPS = {
    "low": 20,
    "medium": 30,
    "high": 40,
    "ultra": 50,
}

ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

def run_text_to_image(positive_prompt: str, negative_prompt: str, image_ratio: str, image_quality: str) -> bytes:
    """
    Run text-to-image processing.

    Args:
        text (str): The text prompt describing the desired image.

    Returns:
        str: a base64-encoded string.
    """
    with no_init_weights():
        transformer = QwenImageTransformer2DModel.from_config(
            QwenImageTransformer2DModel.load_config(
                MODEL_NAME, subfolder="transformer",
            ),
        ).to(torch.bfloat16)

    DFloat11Model.from_pretrained(
        "DFloat11/Qwen-Image-DF11",
        device="cpu",
        cpu_offload=CPU_OFFLOAD,
        cpu_offload_blocks=CPU_OFFLOAD_BLOCKS,
        pin_memory=PIN_MEMORY,
        bfloat16_model=transformer,
    )

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
        "zh": ", 超清，4K，电影级构图." # for chinese prompt
    }

    width, height = ASPECT_RATIOS[image_ratio]
    random_seed = random.randint(0, 999999)
    image = pipe(
        prompt=positive_prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=IMAGE_QUALITY_TO_STEPS[image_quality],
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(random_seed)
    ).images[0]
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    
    return base64.b64encode(image_data)

from processing.generate_image import run_text_to_image, run_image_to_image_edit

from celery import Celery
import os


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

app = Celery(
    "qwen_image_worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)


@app.task(name="text_to_image", queue="qwen_image")
def text_to_image(positive_prompt: str, negative_prompt: str, image_ratio: str, image_quality: str) -> str:
    result = run_text_to_image(positive_prompt=positive_prompt, negative_prompt=negative_prompt, image_ratio=image_ratio, image_quality=image_quality)
    return result.decode('utf-8')


@app.task(name="image_to_image_edit", queue="qwen_image")
def image_to_image_edit(image_bytes: str, positive_prompt: str, negative_prompt: str, image_quality: str) -> str:
    result = run_image_to_image_edit(image_bytes=image_bytes, positive_prompt=positive_prompt, negative_prompt=negative_prompt, image_quality=image_quality)
    return result.decode('utf-8')

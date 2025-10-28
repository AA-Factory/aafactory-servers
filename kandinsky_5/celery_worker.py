from processing.compute_text_to_video import run_text_to_video

from celery import Celery
import os


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

app = Celery(
    "kandinsky_worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)


@app.task(name="kandinsky_text_to_video", queue="kandinsky")
def kandinsky_text_to_video(prompt: str, video_aspect_ratio: str) -> str:
    result = run_text_to_video(prompt=prompt, video_aspect_ratio=video_aspect_ratio)
    return result.decode('utf-8')

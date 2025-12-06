import os
import subprocess


def run_command(cmd_list):
    """Helper to run shell commands and handle errors."""
    try:
        # capture_output=True hides the massive FFmpeg logs unless there is an error
        subprocess.run(cmd_list, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd_list)}")
        print(e.stderr)
        raise e

def generate_subtitles_pipeline(video_path: str, user_args: dict) -> str:
    
    base, ext = os.path.splitext(video_path)
    processed_video_path = f"{base}_processed{ext}"
    data_dir_path = os.path.dirname(video_path)
    srt_path = f"{base}_processed.srt"

    
    # 1. Audio Processing (Denoise + EQ combined)
    print("1. Applying Audio Filters (Denoise + EQ)...")
    filters = "afftdn,highpass=f=200,lowpass=f=8000"
    run_command(["ffmpeg", "-y", "-i", video_path, "-af", filters, processed_video_path])

    # 2. Transcribe (WhisperX)
    print("2. Transcribing with WhisperX...")
    cmd = [
        "whisperx", 
        "--model", "large-v3", 
        "--device", "cuda", 
        "--output_format", "srt",
        "--output_dir", data_dir_path,
        processed_video_path
    ]
    subprocess.call(cmd)

    return srt_path
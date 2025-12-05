import os
import subprocess
import re
import json
import shutil
import google.generativeai as genai
from pydantic import BaseModel
from tqdm import tqdm

# ==========================================
# PART 1: GEMINI TRANSLATION MODULE
# ==========================================

class SubLine(BaseModel):
    id: int
    text: str

class SubBatch(BaseModel):
    subtitles: list[SubLine]

def parse_srt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split('\n\n')
    subs = []
    for block in blocks:
        match = re.match(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.*)', block, re.DOTALL)
        if match:
            subs.append({"id": int(match.group(1)), "time": match.group(2), "text": match.group(3).strip()})
    return subs

def translate_subs(subs: list[dict], api_key: str) -> list[dict]:
    genai.configure(api_key=api_key)
    system_instruction = "You are a professional subtitle translator. Translate the text from Spanish to Serbian (Latin script). Keep the translation concise to fit subtitles. Do not merge lines. Translate it in a way that preserves the meaning while staying true to the Serbian style, rather than doing a literal translation."
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)
    
    batch_size = 20
    print(f"   > Translating {len(subs)} lines via Gemini...")

    for i in tqdm(range(0, len(subs), batch_size)):
        batch = subs[i:i+batch_size]
        payload = [{"id": s["id"], "text": s["text"]} for s in batch]
        try:
            response = model.generate_content(
                json.dumps(payload),
                generation_config=genai.GenerationConfig(response_mime_type="application/json", response_schema=SubBatch)
            )
            translated_data = json.loads(response.text)
            trans_map = {item['id']: item['text'] for item in translated_data['subtitles']}
            for s in batch:
                if s["id"] in trans_map:
                    s["text"] = trans_map[s["id"]]
        except Exception as e:
            print(f"   ! Batch {i} translation failed: {e}")
    return subs

def save_srt(subs: list[dict], output_path: str):
    content = "\n\n".join([f"{s['id']}\n{s['time']}\n{s['text']}" for s in subs])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

# ==========================================
# PART 2: MEDIA PROCESSING PIPELINE
# ==========================================

def run_command(cmd_list):
    """Helper to run shell commands and handle errors."""
    try:
        # capture_output=True hides the massive FFmpeg logs unless there is an error
        subprocess.run(cmd_list, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd_list)}")
        print(e.stderr)
        raise e

def process_single_link(episode_name, link, output_dir, api_key):
    """Runs the full pipeline for a single HLS link."""
    
    # 1. Setup Naming
    raw_mp4 = os.path.join(output_dir, f"{episode_name}_raw.mp4")
    final_mp4 = os.path.join(output_dir, f"{episode_name}.mp4")
    
    # WhisperX output filename prediction (WhisperX appends .srt to the input filename)
    # If we feed it 'job_001_final.mp4', it creates 'job_001_final.srt' inside output_dir
    expected_srt = os.path.join(output_dir, f"{episode_name}.srt")
    translated_srt = os.path.join(output_dir, f"{episode_name}_sr.srt")

    print(f"\n--- Processing {episode_name} ---")

    # 2. Download (FFmpeg)
    if not os.path.exists(raw_mp4):
        print("1. Downloading stream...")
        run_command(["ffmpeg", "-y", "-i", link, "-c", "copy", raw_mp4])
    
    # 3. Audio Processing (Denoise + EQ combined)
    # We combine steps 3 & 4 into one filter chain for efficiency
    if not os.path.exists(final_mp4):
        print("2. Applying Audio Filters (Denoise + EQ)...")
        filters = "afftdn,highpass=f=200,lowpass=f=8000"
        run_command(["ffmpeg", "-y", "-i", raw_mp4, "-af", filters, final_mp4])

    # 4. Transcribe (WhisperX)
    if not os.path.exists(expected_srt):
        print("3. Transcribing with WhisperX (this may take time)...")
        # Note: We use --output_dir to ensure we know where the file lands
        cmd = [
            "whisperx", 
            "--model", "large-v3", 
            "--language", "es", 
            "--device", "cuda", 
            "--output_format", "srt",
            "--output_dir", output_dir,
            final_mp4
        ]
        # We use subprocess.call here because WhisperX prints progress bars we might want to see
        subprocess.call(cmd)

    # 5. Translate (Gemini)
    if os.path.exists(expected_srt) and not os.path.exists(translated_srt):
        print("4. Translating to Serbian...")
        srt_data = parse_srt(expected_srt)
        translated_data = translate_subs(srt_data, api_key)
        save_srt(translated_data, translated_srt)
        print(f"   > Saved: {translated_srt}")
    else:
        print("   ! Error: WhisperX did not generate the expected SRT file.")
        return

    # 6. Cleanup (Optional)
    # Remove the large video files to save space
    print("5. Cleaning up temp files...")
    # if os.path.exists(raw_mp4): os.remove(raw_mp4)
    if os.path.exists(final_mp4): os.remove(final_mp4)
    
    # Optional: Remove the Spanish SRT if you only want the Serbian one
    # if os.path.exists(expected_srt): os.remove(expected_srt)

# ==========================================
# PART 3: MAIN EXECUTION
# ==========================================

def generate_subtitles_pipeline():
    # CONFIGURATION
    API_KEY = "AIzaSyAXTRzF7cRhBnMPLlVfu1kN3mUn7-Upde4"
    OUTPUT_DIR = "pipeline_output"
    
    # List of HLS (.m3u8) links
    links = {
        # "s01e04": "https://vixsrc.to/playlist/328060?token=3d0b2881d276076a08cbac9b3ab93446&expires=1769887664&h=1&lang=en",
        "s01e05": "https://vixsrc.to/playlist/328061?token=15e9040dbfc5f612ca8694c6ba9d8799&expires=1769887902&h=1&lang=en",
    }

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Run Pipeline
    for episode_name, link in links.items():
        try:
            process_single_link(episode_name, link, OUTPUT_DIR, API_KEY)
        except Exception as e:
            print(f"CRITICAL ERROR on {episode_name}: {e}")
            continue

    print("\nAll jobs finished.")

if __name__ == "__main__":
    main()
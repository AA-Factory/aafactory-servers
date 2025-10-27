#!/bin/bash
set -euo pipefail

# ---- ComfyUI model downloader ----
BASE_DIR="/app/comfyui_logic/ComfyUI/models"
CURL_BIN="${CURL_BIN:-curl}"

declare -A MODELS=(
  ["$BASE_DIR/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors"]="https://huggingface.co/chatpig/encoder/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  ["$BASE_DIR/clip_vision/clip_vision_h.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
  ["$BASE_DIR/diffusion_models/Wan2_2-Animate-14B_fp8_e5m2_scaled_KJ.safetensors"]="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_e5m2_scaled_KJ.safetensors"
  ["$BASE_DIR/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"]="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
  ["$BASE_DIR/loras/WanAnimate_relight_lora_fp16.safetensors"]="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"
  ["$BASE_DIR/sam2/sam2_hiera_base_plus.safetensors"]="https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_hiera_base_plus.safetensors"
  ["$BASE_DIR/sams/sam_vit_b_01ec64.pth"]="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth"
  ["$BASE_DIR/vae/Wan2_1_VAE_bf16.safetensors"]="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"
)

for dest in "${!MODELS[@]}"; do
  url="${MODELS[$dest]}"
  dir="$(dirname "$dest")"
  if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
  fi
  tmp="${dest}.part"
  echo "Downloading: $(basename "$dest") -> $dest"
  $CURL_BIN --fail --location --continue-at - --retry 5 --retry-delay 5 --output "$tmp" "$url"
  mv -f "$tmp" "$dest"
  chmod 644 "$dest" || true
done

echo "Model weights downloaded. Starting Celery worker..."
redis-server --protected-mode no &
uv run celery -A celery_worker.app worker --loglevel=info -Q animate

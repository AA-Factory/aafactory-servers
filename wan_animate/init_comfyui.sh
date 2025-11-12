#!/bin/bash
set -e

BASE_DIR="/app/wan_animate"
COMFYUI_DIR="$BASE_DIR/ComfyUI"
CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"

# Clone ComfyUI repo to specific commit
echo "Cloning ComfyUI..."
mkdir -p "$BASE_DIR"
git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
git -C "$COMFYUI_DIR" checkout 388b306a2b48070737b092b51e76de933baee9ad

echo "Cloning custom nodes..."
mkdir -p "$CUSTOM_NODES_DIR"
cd "$CUSTOM_NODES_DIR" && \
    git clone https://github.com/evanspearman/ComfyMath.git && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/MixLabPro/comfyui-mixlab-nodes.git && \
    git clone https://github.com/kijai/ComfyUI-segment-anything-2.git && \
    git clone https://github.com/un-seen/comfyui-tensorops.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    git clone https://github.com/rgthree/rgthree-comfy.git
    
echo "Pinning custom nodes to specific commits..."
git -C "$CUSTOM_NODES_DIR/ComfyMath" checkout c01177221c31b8e5fbc062778fc8254aeb541638 && \
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Custom-Scripts" checkout f2838ed5e59de4d73cde5c98354b87a8d3200190 && \
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Easy-Use" checkout 125b3a4905ac9367f7aae5c2df67e2282f0a3d37 && \
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Frame-Interpolation" checkout a969c01dbccd9e5510641be04eb51fe93f6bfc3d && \
    git -C "$CUSTOM_NODES_DIR/ComfyUI-KJNodes" checkout 6ee278aa7d802c03ad713a7c5fbc02861b6773a8 && \
    git -C "$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite" checkout 6e7f63867584bb1fb4944d36172e7f98436da9a1 && \
    git -C "$CUSTOM_NODES_DIR/ComfyUI-WanVideoWrapper" checkout d74cfc54e8ae065d7f32b36634be770bae525f5c && \
    git -C "$CUSTOM_NODES_DIR/ComfyUI-segment-anything-2" checkout 0c35fff5f382803e2310103357b5e985f5437f32 && \
    git -C "$CUSTOM_NODES_DIR/comfyui-mixlab-nodes" checkout 32b22c39cbe13b46df29ef1b6ab088c2eb4389d2 && \
    git -C "$CUSTOM_NODES_DIR/comfyui-tensorops" checkout d34488e3079ecd10db2fe867c3a7af568115faed && \
    git -C "$CUSTOM_NODES_DIR/comfyui_controlnet_aux" checkout 12f35647f0d510e03b45a47fb420fe1245a575df && \
    git -C "$CUSTOM_NODES_DIR/rgthree-comfy" checkout 2b9eb36d3e1741e88dbfccade0e08137f7fa2bfb

echo "Setup complete."

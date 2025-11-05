#!/bin/bash
set -e

# Clone ComfyUI repo to specific commit
mkdir -p /app/comfyui_logic
cd /app/comfyui_logic && git clone https://github.com/comfyanonymous/ComfyUI.git && \
    git -C /app/comfyui_logic/ComfyUI checkout 388b306a2b48070737b092b51e76de933baee9ad

# Install all custom nodes (clone)
cd /app/comfyui_logic/ComfyUI/custom_nodes && \
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
# Pin each cloned repo to the specific commit
git -C /app/comfyui_logic/ComfyUI/custom_nodes/ComfyMath checkout c01177221c31b8e5fbc062778fc8254aeb541638 && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts checkout f2838ed5e59de4d73cde5c98354b87a8d3200190 && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/ComfyUI-Easy-Use checkout 125b3a4905ac9367f7aae5c2df67e2282f0a3d37 && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation checkout a969c01dbccd9e5510641be04eb51fe93f6bfc3d && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/ComfyUI-KJNodes checkout 6ee278aa7d802c03ad713a7c5fbc02861b6773a8 && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite checkout 6e7f63867584bb1fb4944d36172e7f98436da9a1 && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper checkout d74cfc54e8ae065d7f32b36634be770bae525f5c && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/ComfyUI-segment-anything-2 checkout 0c35fff5f382803e2310103357b5e985f5437f32 && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/comfyui-mixlab-nodes checkout 32b22c39cbe13b46df29ef1b6ab088c2eb4389d2 && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/comfyui-tensorops checkout d34488e3079ecd10db2fe867c3a7af568115faed && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/comfyui_controlnet_aux checkout 12f35647f0d510e03b45a47fb420fe1245a575df && \
    git -C /app/comfyui_logic/ComfyUI/custom_nodes/rgthree-comfy checkout 2b9eb36d3e1741e88dbfccade0e08137f7fa2bfb

# Workflow Edits

If `workflow.py` is changed, before testing make sure to:

## 1. Double check if constants in workflow match desired constants for input/output, config and models
1. For each value in `wan_animate/arg_config.py ARG_CONFIG` dictionary, check if its corresponding `map_to` field is matching same name in `wan_animate/workflow.py`.
2. Double check if you have correct model names (some models are for older GPU generations `e.g. Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors -> Wan2_2-Animate-14B_fp8_e5m2_scaled_KJ.safetensors`)
* Best practive is to replace with old workflow.py and see git diff
# Workflow Edits

## 1. Double check if constants in workflow match desired constants for input/output, config and models
1. If you are passing any configs when calling the script, check if the names of the variables have changed
2. Double check if you have correct model names (some models are for older GPU generations `e.g. Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors -> Wan2_2-Animate-14B_fp8_e5m2_scaled_KJ.safetensors`)
* Best practive is to replace with old workflow.py and see git diff
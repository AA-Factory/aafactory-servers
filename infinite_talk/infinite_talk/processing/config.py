# Default arguments
default_args = {
        "ckpt_dir": "weights/Wan2.1-I2V-14B-480P", 
        "wav2vec_dir": "weights/chinese-wav2vec2-base",
        "infinitetalk_dir": "weights/InfiniteTalk/single/infinitetalk.safetensors",
        "input_json": "single_example_image.json",
        "audio_save_dir": "save_audio",
        "dit_path": None,
        "quant_dir": "weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors",
        "lora_dir": None,
        "lora_scale": 1.2,
        "base_seed": 42,
        "size": "infinitetalk-480",
        "sample_steps": 20,
        "sample_shift": 3.0,
        "sample_text_guide_scale": 5.0,
        "sample_audio_guide_scale": 4.0,
        "num_persistent_param_in_dit": None,
        "audio_mode": "localfile",
        "mode": "streaming",
        "motion_frame": 9,
        "offload_model": None,
        "t5_fsdp": None,
        "t5_cpu": False,
        "dit_fsdp": None,
        "use_teacache": False,
        "teacache_thresh": 0.2,
        "ulysses_size": 1,
        "ring_size": 1,
        "use_apg": False,
        "apg_momentum": -0.75,
        "apg_norm_threshold": 55,
        "color_correction_strength": 1.0,
        "scene_seg": False,
        "quant": None,
        "task": "infinitetalk-14B",
        "frame_num": 81,
        "max_frame_num": 100000,
        "save_file": "infinitetalk_res_quant"
}

    
# Configuration overrides
config_overrides = {
    "8_steps": {  # LoRA
        "lora_dir": ["weights/Wan14BT2VFusioniX/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"],          
        "lora_scale": [1.0],
        "sample_text_guide_scale": 1.0,
        "sample_audio_guide_scale": 2.0,
        "sample_steps": 8,
        "sample_shift": 2,
        "save_file": "infinitetalk_res_lora"
    },
    "6_steps": {  # LoRA
        "lora_dir": ["weights/WanVideo_comfy/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"],
        "lora_scale": [1.0],
        "sample_text_guide_scale": 1.0,
        "sample_audio_guide_scale": 2.0,
        "sample_steps": 6,
        "sample_shift": 2,
        "save_file": "infinitetalk_res_lora"
    },
    "high_quality": {  # 720p            
        "size": "infinitetalk-720",
        "sample_steps": 40,
        "save_file": "infinitetalk_res_720p"
    },
    "medium_quality": {  # medium quality            
        "sample_steps": 20,
        "save_file": "infinitetalk_res"
    },
    "quantize_model": {  # Quantized            
        "sample_steps": 20,
        "quant": "fp8",
        "save_file": "infinitetalk_res_quant"
    }
}

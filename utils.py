import torch
import GPUtil
import math
import random
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, \
    FluxPipeline, StableDiffusion3Pipeline

MAX_SEED = 2**32 - 1
TEXT_TO_IMAGE_DICTIONARY = {
    # "FLUX.1 [schnell-dev-merged]": {
    #     "path": "sayakpaul/FLUX.1-merged",
    #     "pipeline": FluxPipeline,
    #     "device_map": "balanced"
    # },
    # "FLUX.1 [schnell]": {
    #     "path": "black-forest-labs/FLUX.1-schnell",
    #     "pipeline": FluxPipeline,
    #     "device_map": "balanced"
    # },
    # "FLUX.1 [dev]": {
    #     "path": "black-forest-labs/FLUX.1-dev",
    #     "pipeline": FluxPipeline,
    #     "device_map": "balanced"
    # },
    # "Stable Diffusion 2.1": {
    #     "path": "stabilityai/stable-diffusion-2-1",
    #     "pipeline": StableDiffusionPipeline
    # },
    "Stable Diffusion 3.5 Medium": {
        "backend": "comfyui",
        "path": "stuffs/comfyui_workflow_api/sd3_5_workflow_api.json",
        "device_map": "balanced"
    },
    "Stable Diffusion 3.5 Large": {
        "backend": "comfyui",
        "path": "stuffs/comfyui_workflow_api/sd3_5_workflow_api.json",
        "device_map": "balanced"
    },
    # "Stable Diffusion 3 Medium": {
    #     "path": "stabilityai/stable-diffusion-3-medium-diffusers",
    #     "pipeline": StableDiffusion3Pipeline,
    #     "device_map": "balanced"
    # },
    # "Realistic SDXL": {
    #     "path": "misri/epicrealismXL_v7FinalDestination",
    #     "pipeline": StableDiffusionXLPipeline,
    # },
    # "DreamShaper8 (SD 1.5)": {
    #     "path": "Lykon/dreamshaper-8",
    #     "pipeline": StableDiffusionPipeline,
    # },
    # "Anime (SD 1.5)": {
    #     "path": "../checkpoints/darkSushiMixMix_225D.safetensors",
    #     "pipeline": StableDiffusionPipeline,
    # }
}


def nearest_divisible_by_8(number: int = 1024):
    """
    Auto adjust the number to make it divisible by 8
    """
    lower_multiple = (number // 8) * 8
    upper_multiple = lower_multiple + 8
    if (number - lower_multiple) < (upper_multiple - number):
        return int(lower_multiple)
    else:
        return int(upper_multiple)


def get_gpu_info(width: int = 1024, 
                 height: int = 1024,
                 num_images: int = 1) -> tuple[list, dict]:
    """
    Get available GPUs info

    Parameters:
        - width : generated image's width
        - height : generated image's height
        - num_images : number of generated images per prompt
    Returns:
        - gpu_info and current_max_memory
    """
    gpus = GPUtil.getGPUs()
    gpu_info = []
    current_max_memory = {}
    using_fast_flux = width <= 1280 \
              and height <= 1280 \
              and num_images==1
    for gpu in gpus:
        info = {
            'id': gpu.id,
            'name': gpu.name,
            'driver_version': gpu.driver,
            'total_memory': gpu.memoryTotal,  # In MB
            'available_memory': gpu.memoryFree,  # In MB
            'used_memory': gpu.memoryUsed,  # In MB
            'temperature': gpu.temperature  # In Celsius
        }
        gpu_info.append(info)
        if using_fast_flux:
            current_max_memory[gpu.id] = f"{math.ceil(gpu.memoryFree / 1024)}GB"
        else:
            current_max_memory[gpu.id] = f"{int(gpu.memoryFree / 1024)}GB"


    return gpu_info, current_max_memory


def generate_number():
    """
    Random an integer

    Returns:
        - int: an integer
    """
    return random.randint(0, MAX_SEED)


def assign_gpu(required_vram, width, height, num_images):
    """
    Assign GPU device
    
    Parameters:
        - required_memory (int): minimum VRAM
    
    Returns:
        - torch.device
    """
    gpu_info, _ = get_gpu_info(width, height, num_images)
    device = "cpu"
    for gpu in gpu_info:
        if gpu['available_memory'] >= required_vram:
            device = f"cuda:{gpu['id']}"
    if device == "cpu":
        return device
    return torch.device(device)


# def optimize_flux_pipeline(width, height, num_images):
#     """
    

#     Parameters:
#         - width (int): generated image width
#         - height (int): generated image height
#         - num_images (int): number of generated images per prompt
#     """
#     using_fast_flux = width <= 1280 \
#               and height <= 1280 \
#               and num_images==1
#     return using_fast_flux
import gradio as gr
import GPUtil
import random
import string
import re
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline

DIFFUSION_CHECKPOINTS = {
    "General (SD 3 Medium)": {
        "path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "type": "pretrained",
        "pipeline": StableDiffusion3Pipeline,
        "half_precision": True,
    },
    "Anime (SD 1.5)": {
        "path": "../checkpoints/darkSushiMixMix_225D.safetensors",
        "type": "file",
        "pipeline": StableDiffusionPipeline
    },
    "Anime AnyLoRA (SD 1.5)": {
        "path": "../checkpoints/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors",
        "type": "file",
        "pipeline": StableDiffusionPipeline
    },
    "Cartoon (SD 1.5)": {
        "path": "../checkpoints/animesh_FullV22.safetensors",
        "type": "file",
        "pipeline": StableDiffusionPipeline
    },
    "Realistic (SD 1.5)": {
        "path": "../checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors",
        "type": "file",
        "pipeline": StableDiffusionPipeline
    },
    "Realistic Asian (SD 1.5)": {
        "path": "../checkpoints/majicmixRealistic_v7.safetensors",
        "type": "file",
        "pipeline": StableDiffusionPipeline
    },
    "Realistic XL (SDXL 1.0)": {
        "path": "../checkpoints/epicrealismXL_v7FinalDestination.safetensors",
        "type": "file",
        "pipeline": StableDiffusionXLPipeline,
    }
}
USERS = {
    "admin":
    {
        "password": "admin",
    }
}
AUTH_USERS = [(username, USERS[username]["password"]) for username in USERS.keys()]
APP_THEME = gr.Theme.from_hub("ParityError/Interstellar")
# APP_THEME = gr.Theme.from_hub("EveryPizza/Cartoony-Gradio-Theme")

def generate_random_string(length=8):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def add_user(username, password):
    global USERS
    return

def read_md_file_to_string(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        return file_content
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def nearest_divisible_by_8(n):
    lower_multiple = (n // 8) * 8
    upper_multiple = lower_multiple + 8
    if (n - lower_multiple) < (upper_multiple - n):
        return int(lower_multiple)
    else:
        return int(upper_multiple)

def get_gpu_info():
    gpus = GPUtil.getGPUs()
    gpu_info = []
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
    return gpu_info

def display_gpu_info():
    info_list = []
    gpus = get_gpu_info()
    for info in gpus:
        info_list.append(f"GPU {info['id']} ({info['name']}, Total: {info['total_memory']} MB, Available: {info['available_memory']} MB)")
    return info_list

def find_lora_scale(tag: str = ''):
    pattern = r"<lora_scale:(0\.\d+)>"
    match = re.search(pattern, tag)
    if match:
        # Extract the number 0.85
        number = match.group(1)
        number = float(number)
        if 0 <= number <= 1:
            return number
        else:
            return 1
    else:
        return 1

tonai_creative_html = read_md_file_to_string("stuffs/html/tonai_creative_info.html")
# tonai_chat_html = read_md_file_to_string("stuffs/html/tonai_chat.html")
home_header_html = read_md_file_to_string("stuffs/html/homepage.html")
with open("stuffs/tips.md") as txtfile:
    tips_content = txtfile.read()

custom_css = """
<style>
.gradio input[type="file"][data-label="LoRA safetensors file"] {
    height: 50px !important;
    width: 300px !important;
}
</style>
"""
# js_func = """
# function refresh() {
#     const url = new URL(window.location);

#     if (url.searchParams.get('__theme') !== 'light') {
#         url.searchParams.set('__theme', 'light');
#         window.location.href = url.href;
#     }
# }
# """
import gradio as gr
import GPUtil
import random
import string

DIFFUSION_CHECKPOINTS = {
    "General (SD 3 Medium)": {
        "path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "type": "pretrained",
        "pipeline": "StableDiffusion3Pipeline"
    },
    # "Realistic (SD 1.5)": {
    #     "path": "/root/tungn197/genAI/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors",
    #     "type": "file",
    #     "pipeline": "StableDiffusionPipeline"
    # },
    "Anime (SD 1.5)": {
        "path": "/root/tungn197/genAI/checkpoints/darkSushiMixMix_225D.safetensors",
        "type": "file",
        "pipeline": "StableDiffusionPipeline"
    },
    "Comic Book (SD 1.5)": {
        "path": "/media/drive-2t/tungn197/checkpoints/realisticComicBook_v10.safetensors",
        "type": "file",
        "pipeline": "StableDiffusionPipeline"
    },
    "Realistic Asian (SD 1.5)": {
        "path": "/root/tungn197/genAI/checkpoints/majicmixRealistic_v7.safetensors",
        "type": "file",
        "pipeline": "StableDiffusionPipeline"
    },
    # "ChilloutMix (SD 1.5)": {
    #     "path": "/media/drive-2t/tungn197/checkpoints/ChilloutMix.safetensors",
    #     "type": "file",
    #     "pipeline": "StableDiffusionPipeline",
    #     "lora": "/root/tungn197/genAI/checkpoints/lora_yangmiV73-000006.safetensors"
    # },
    "AniMeshFullV22 (SD 1.5, Cartoon style)": {
        "path": "/media/drive-2t/tungn197/checkpoints/animesh_FullV22.safetensors",
        "type": "file",
        "pipeline": "StableDiffusionPipeline"
    },
    "epiCRealism XL (SDXL 1.0, Realistic style)": {
        "path": "/root/tungn197/genAI/checkpoints/epicrealismXL_v7FinalDestination.safetensors",
        "type": "file",
        "pipeline": "StableDiffusionXLPipeline",
        "lora": "/root/tungn197/genAI/checkpoints/mod2.safetensors"
    },
    # "Juggernaut X Hyper (SDXL 1.0)": {
    #     "path": "RunDiffusion/Juggernaut-X-Hyper",
    #     "type": "pretrained",
    #     "pipeline": "StableDiffusionXLPipeline"
    # }
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

tonai_creative_html = read_md_file_to_string("stuffs/html/tonai_creative_info.html")
tonai_chat_html = read_md_file_to_string("stuffs/html/tonai_chat.html")
home_header_html = read_md_file_to_string("stuffs/html/homepage.html")
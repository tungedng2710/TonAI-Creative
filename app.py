import os
import time
import random
import torch
import gc
import gradio as gr
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from accelerate import PartialState
# from scheduler_mapping import schedulers, get_scheduler
from utils import *


def gen_image(prompt, negative_prompt, width, height,
              num_steps, mode, seed, guidance_scale,
              lora_weight_file):
    """
    Run diffusion model to generate image
    """
    distributed_state = PartialState()
    use_lora = False
    available_gpus = get_gpu_info()
    guidance_scale = float(guidance_scale)
    model_path = DIFFUSION_CHECKPOINTS[mode]["path"]
    Text2Image_class = DIFFUSION_CHECKPOINTS[mode]["pipeline"]
    Text2Image_class.safety_checker = None
    if DIFFUSION_CHECKPOINTS[mode]["type"] == "pretrained":
        if DIFFUSION_CHECKPOINTS[mode]["half_precision"]:
            pipeline = Text2Image_class.from_pretrained(
                model_path, torch_dtype=torch.float16, use_safetensors=True)
        else:
            pipeline = Text2Image_class.from_pretrained(
                model_path, device_map="balanced", use_safetensors=True)
    else:
        pipeline = Text2Image_class.from_single_file(
            model_path, device_map="balanced")

    # pipeline.enable_model_cpu_offload()
    if DIFFUSION_CHECKPOINTS[mode]["pipeline"] is not StableDiffusion3Pipeline:
        # DPM++ 2M SDE Karras
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++")
    if lora_weight_file is not None:
        directory, file_name = os.path.split(lora_weight_file.name)
        try:
            print("LoRA weight was found, trying to load...")
            pipeline.load_lora_weights(
                directory,
                weight_name=file_name,
                adapter_name=file_name.replace(".safetensors",''))
            print("LoRA weight loaded succesfully")
            use_lora = True
        except Exception as e:
            print(e)
            print("Cannot load LoRA weight")
            pass
    image = Image.open("stuffs/serverdown.png")
    # time.sleep(5) # Delay 5 seconds
    for counter, gpu in enumerate(available_gpus):
        if (
                "SDXL" in mode or "SD 3" in mode) and gpu['available_memory'] < 16384:
            if "SD 3" in mode and counter == (len(available_gpus) - 1):
                for gpu in available_gpus:
                    if gpu['available_memory'] > 10000:
                        # Dropping the T5 Text Encoder during Inference if not
                        # enough GPU memory
                        print(
                            "Not enough GPU memory for Stable Diffusion 3, trying to drop T5 Text encoder")
                        pipeline.text_encoder_3 = None
                        pipeline.tokenizer_3 = None
                        break
            else:
                torch.cuda.empty_cache()
                gc.collect()
                continue
        device = torch.device(f"cuda:{gpu['id']}")
        generator = torch.Generator("cuda").manual_seed(int(seed))
        try:
            # use tag <lora_scale:[number in (0, 1)]>
            if use_lora:
                cross_attention_kwargs = {"scale": find_lora_scale(prompt)}
            else:
                cross_attention_kwargs = {}
            pipeline = pipeline.to(device)
            pipeline_configs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": nearest_divisible_by_8(int(width)),
                "height": nearest_divisible_by_8(int(height)),
                "num_inference_steps": int(num_steps),
                "generator": generator,
                "guidance_scale": guidance_scale
            }
            if "SD 3" not in mode:
                pipeline_configs = dict(pipeline_configs, **cross_attention_kwargs)
            image = pipeline(**pipeline_configs).images[0]
            break
        except Exception as e:
            print(f"Exception: {e}")
            print("Not enough GPU memory, trying to change device")
            if counter < (len(available_gpus) - 1):
                continue
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    return image


with gr.Blocks(title="TonAI Creative", theme=APP_THEME, css=custom_css) as interface:
    gr.HTML(tonai_creative_html)
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate")
            with gr.Row():
                width = gr.Textbox(label="Image Width", value=1024, scale=2)
                height = gr.Textbox(label="Image Height", value=1024, scale=2)
                mode = gr.Dropdown(
                    choices=DIFFUSION_CHECKPOINTS.keys(),
                    label="Mode",
                    value=list(
                        DIFFUSION_CHECKPOINTS.keys())[0],
                    interactive=True,
                    scale=4)
                generate_btn = gr.Button("Generate", scale=2)
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value='',
                    placeholder="Instruct the AI model that it should not include")
                with gr.Row():
                    seed = gr.Textbox(label="RNG Seed", value=0, scale=1)
                    guidance_scale = gr.Textbox(
                        label="CFG Scale", value=7.5, scale=1)
                    num_steps = gr.components.Slider(
                        minimum=5, maximum=60, value=23, step=1,
                        label="Inference Steps",
                        scale=3
                    )
                lora_weight_file = gr.File(
                    label="LoRA safetensors file",
                    elem_classes=["file-input"])
            with gr.Accordion("Helps", open=False):
                gr.Markdown(tips_content)

        with gr.Column(scale=1):
            generate_btn.click(
                fn=gen_image,
                inputs=[
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    num_steps,
                    mode,
                    seed,
                    guidance_scale,
                    lora_weight_file],
                outputs=gr.Image(
                    label="Generated Image",
                    format="png"),
                concurrency_limit=10)
        interface.load(
            lambda: gr.update(
                value=random.randint(
                    0, 999999)), None, seed)

if __name__ == '__main__':
    allowed_paths = ["stuffs/tonai_research_logo.png"]
    interface.queue(default_concurrency_limit=10)
    interface.launch(share=True,
                     root_path="/tonai",
                     server_name=None,
                     # auth=AUTH_USERS,
                     allowed_paths=allowed_paths,
                     max_threads=10)

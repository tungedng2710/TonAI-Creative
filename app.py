import os
import time
import json
import random
import torch
import gc
import gradio as gr
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
# from scheduler_mapping import schedulers, get_scheduler
from utils import *


def gen_image(prompt, negative_prompt, width, height,
              num_steps, mode, seed, guidance_scale,
              lora_weight_file, fp16=False):
    """
    Run diffusion model to generate image
    """
    # distributed_state = PartialState()
    num_images = 4
    model = DIFFUSION_CHECKPOINTS[mode]
    use_lora = False
    available_gpus, current_max_memory = get_gpu_info()
    guidance_scale = float(guidance_scale)
    Text2Image_class = model["pipeline"]
    Text2Image_class.safety_checker = None
    diffusion_configs = {
        "use_safetensors": True,
        "device_map": "balanced",
        "max_memory": current_max_memory
    }
    if fp16:
        diffusion_configs["torch_dtype"] = torch.float16

    if model["type"] == "pretrained":
        pipeline = Text2Image_class.from_pretrained(
            model["path"], **diffusion_configs)
    else:
        diffusion_configs["device_map"] = "auto"
        pipeline = Text2Image_class.from_single_file(
            model["path"], **diffusion_configs)

    if model["pipeline"] is not StableDiffusion3Pipeline:
        # DPM++ 2M SDE Karras
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++")
    
    # Load LoRA adapter
    if lora_weight_file is not None:
        directory, file_name = os.path.split(lora_weight_file.name)
        try:
            print("LoRA weight was found, trying to load...")
            pipeline.load_lora_weights(
                directory,
                weight_name=file_name,
                adapter_name=file_name.replace(".safetensors", ''))
            print("LoRA weight loaded succesfully")
            use_lora = True
        except Exception as e:
            print(e)
            print("Cannot load LoRA weight")
            pass
    images = [Image.open("stuffs/serverdown.png")]
    time.sleep(5)  # Delay 5 seconds
    for counter, gpu in enumerate(available_gpus):
        if "SD 3" in mode and gpu['available_memory'] < 10000:
            if not fp16:
                # Drop T5 encoder to reduce memory usage
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
            # remove lora tag if it exists
            prompt = re.sub(r'<.*?>', '', prompt)
            # pipeline = pipeline.to(device)
            prompt = [prompt] * num_images # Generate multiple images
            negative_prompt = [negative_prompt] * num_images
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
                pipeline = pipeline.to(device)
                pipeline_configs["cross_attention_kwargs"] = cross_attention_kwargs
            images = pipeline(**pipeline_configs).images
            break
        except Exception as e:
            print(f"Exception: {e}")
            if counter < (len(available_gpus) - 1):
                continue
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    return images


# -------------------------------------------- Gradio App -------------------------------------------- #
with gr.Blocks(title="TonAI Creative", theme=APP_THEME, css=custom_css) as interface:
    gr.HTML(tonai_creative_html)
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Accordion("Basic Usage", open=True):
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate")
                with gr.Row():
                    width = gr.Textbox(
                        label="Image Width", value=1024, scale=1)
                    height = gr.Textbox(
                        label="Image Height", value=1024, scale=1)
                    mode = gr.Dropdown(
                        choices=DIFFUSION_CHECKPOINTS.keys(),
                        label="Mode",
                        value=list(
                            DIFFUSION_CHECKPOINTS.keys())[0],
                        interactive=True,
                        scale=2)
                    fp16 = gr.Checkbox(
                        label="Fast Inference",
                        info="Faster run but decrease picture quality a bit",
                        scale=1)
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

        with gr.Column(scale=3):
            gallery = gr.Gallery(
                label="Generated images",
                elem_id="gallery",
                columns=2,
                rows=2,
                object_fit="fill")
            click_button_behavior = {
                "fn": gen_image,
                "outputs": gallery,
                "concurrency_limit": 10
            }
            generate_btn.click(inputs=[prompt,
                                       negative_prompt,
                                       width,
                                       height,
                                       num_steps,
                                       mode,
                                       seed,
                                       guidance_scale,
                                       lora_weight_file,
                                       fp16],
                               **click_button_behavior)
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
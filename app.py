import os
import random
import torch
import gc
import gradio as gr
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler,\
                      StableDiffusionXLPipeline, StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from utils import *

def gen_image(prompt, negative_prompt, width, height, num_steps, mode, seed, guidance_scale):
    """
    Run diffusion model to generate image
    """
    available_gpus = get_gpu_info()
    guidance_scale = float(guidance_scale)
    model_path = DIFFUSION_CHECKPOINTS[mode]["path"]
    Text2Image_class = globals()[DIFFUSION_CHECKPOINTS[mode]["pipeline"]]
    Text2Image_class.safety_checker=None
    if DIFFUSION_CHECKPOINTS[mode]["type"] == "pretrained":
        if DIFFUSION_CHECKPOINTS[mode]["pipeline"] == "StableDiffusion3Pipeline": # half precision for fp16
            pipeline = Text2Image_class.from_pretrained(model_path,
                                                        text_encoder_3=None,
                                                        tokenizer_3=None,
                                                        torch_dtype=torch.float16)
        else:
            pipeline = Text2Image_class.from_pretrained(model_path)
    else:
        pipeline = Text2Image_class.from_single_file(model_path)

    if "lora" in DIFFUSION_CHECKPOINTS[mode]:
        directory, filename = os.path.split(DIFFUSION_CHECKPOINTS[mode]["lora"])
        pipeline.load_lora_weights(directory, weight_name=filename)
    # pipeline.enable_model_cpu_offload()
    if DIFFUSION_CHECKPOINTS[mode]["pipeline"] == "StableDiffusion3Pipeline":
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    image = Image.open("stuffs/serverdown.png")
    for counter, gpu in enumerate(available_gpus):
        if "SDXL" in mode and gpu['available_memory'] < 15000:
            continue
        device = torch.device(f"cuda:{gpu['id']}")
        generator = torch.Generator(device).manual_seed(int(seed))
        try:
            pipeline = pipeline.to(device)
            image = pipeline(prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=nearest_divisible_by_8(int(width)),
                            height=nearest_divisible_by_8(int(height)),
                            num_inference_steps=int(num_steps),
                            generator=generator,
                            guidance_scale=guidance_scale).images[0]
            break
        except Exception as e:
            if counter < (len(available_gpus) - 1):
                continue
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    return image

with gr.Blocks(title="TonAI Creative", theme=APP_THEME) as interface:
    gr.HTML(tonai_creative_html)
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate")
            with gr.Row():
                width = gr.Textbox(label="Image Width", value=1024)
                height = gr.Textbox(label="Image Height", value=1024)
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Textbox(label="Negative Prompt", value='', placeholder="Instruct the AI model that it should not include")
                with gr.Row():
                    seed = gr.Textbox(label="RNG Seed", value=0, scale=1)
                    guidance_scale = gr.Textbox(label="CFG Scale", value=7.5, scale=1)
                with gr.Row():
                    num_steps = gr.components.Slider(
                                    minimum=5, maximum=60, value=23, step=1,
                                    label="Inference Steps"
                                )
                    mode=gr.Dropdown(choices=DIFFUSION_CHECKPOINTS.keys(), label="Mode",
                                     value=list(DIFFUSION_CHECKPOINTS.keys())[0])
                device_choices = display_gpu_info()
                # device=gr.Dropdown(choices=device_choices, label="Device", value=device_choices[0])
            generate_btn = gr.Button("Generate")
        with gr.Column(scale=2):
            generate_btn.click(
                        fn=gen_image,
                        inputs=[prompt, negative_prompt, width, height, num_steps, mode, seed, guidance_scale],
                        outputs=gr.Image(label="Generated Image", format="png"),
                        concurrency_limit=10
                    )
        interface.load(lambda: gr.update(value=random.randint(0, 999999)), None, seed)
        # interface.load(lambda: gr.update(choices=display_gpu_info(), value=display_gpu_info()[0]), None, device)

if __name__ == '__main__':
    allowed_paths=["stuffs/tonai_research_logo.png"]
    interface.queue(default_concurrency_limit=10)
    interface.launch(share=True,
                     root_path="/tonai",
                     server_name=None,
                     server_port=7860,
                     # auth=AUTH_USERS,
                     allowed_paths=allowed_paths,
                     max_threads=10)
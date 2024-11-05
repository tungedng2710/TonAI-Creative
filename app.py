import os
import random
import torch
import gc
import gradio as gr
import style as sty
from PIL import Image
from scheduler_mapping import schedulers, apply_scheduler
from utils import *

from diffusers.utils import logging
from query_comfyui import *

logging.set_verbosity_info()
logging.get_logger("diffusers").setLevel(logging.ERROR)

SCHEDULERS = list(schedulers.keys())
SCHEDULERS.insert(0, "Default")


def gen_image(prompt, negative_prompt, width, height,
              num_steps, mode, seed, guidance_scale,
              lora_weight_file, lora_scale, fast_infer,
              scheduler, num_images, progress=gr.Progress(track_tqdm=True)):
    """
    Run diffusion model to generate image
    """
    progress(0, "Starting image generation...")
    for i in range(1, num_steps + 1):
        progress(i / num_steps * 100, f"Processing step {i} of {num_steps}...")

    images = [Image.open("stuffs/logo.png")]
    
    if len(prompt) == 0:
        gr.Info("Please input prompt!", duration=5)
        return images
    
    # Query COmfyUI backend
    if "Stable Diffusion 3.5" in mode:
        if "Medium" in mode:
            ckpt_name = "sd3.5_medium.safetensors"
        else:
            ckpt_name = "sd3.5_large.safetensors"
        images = query_sd35(ckpt_name, prompt, negative_prompt,
                            int(width), int(height),
                            int(num_images), int(seed),
                            float(guidance_scale), int(num_steps))
        return images
    
    model = TEXT_TO_IMAGE_DICTIONARY[mode]
    use_lora = False
    _, current_max_memory = get_gpu_info(width, height, num_images)
    Text2Image_class = model["pipeline"]

    diffusion_configs = {
        "use_safetensors": True,
        "max_memory": current_max_memory
    }
    if "device_map" in model:
        diffusion_configs["device_map"] = model["device_map"]
    if fast_infer:
        diffusion_configs["torch_dtype"] = torch.float16
    if "FLUX" in mode:
        diffusion_configs["torch_dtype"] = torch.bfloat16


    if model["path"].endswith('.safetensors'):
        pipeline = Text2Image_class.from_single_file(
            model["path"], **diffusion_configs)
    else:
        pipeline = Text2Image_class.from_pretrained(
            model["path"], **diffusion_configs)
    pipeline.safety_checker = None

    try:
        pipeline = apply_scheduler(scheduler, pipeline)
    except BaseException:
        gr.Warning(f"Cannot apply {scheduler} for {mode}. Use default sampler instead")
        pipeline = apply_scheduler("Default", pipeline)
        
    # Load LoRA adapter
    if lora_weight_file is not None:
        directory, file_name = os.path.split(lora_weight_file.name)
        try:
            pipeline.load_lora_weights(
                directory,
                weight_name=file_name,
                adapter_name=file_name.replace(".safetensors", ''))
            gr.Info("LoRA weight loaded succesfully", duration=5)
            use_lora = True
        except Exception as e:
            print(e)
            gr.Warning("Cannot load LoRA weight, your model won't use adapter", duration=5)

    # Assign GPU for pipeline
    # if "FLUX" not in mode and "Stable Diffusion 3" not in mode:
    device = assign_gpu(required_vram=10000,
                        width=width,
                        height=height,
                        num_images=num_images)
    if device == "cpu":
        gr.Warning("No available GPUs for inference")
        return images

    generator = torch.Generator("cuda").manual_seed(int(seed))
    try:
        pipeline_configs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": nearest_divisible_by_8(int(width)),
            "height": nearest_divisible_by_8(int(height)),
            "num_inference_steps": int(num_steps),
            "generator": generator,
            "guidance_scale": float(guidance_scale),
            "num_images_per_prompt": num_images
        }
        if "FLUX" not in mode:
            pipeline = pipeline.to(device)
        else:
            # Adjust for FLUX Pipeline
            del pipeline_configs["negative_prompt"]
            # Max 256 tokens for prompt
            pipeline_configs["max_sequence_length"] = 256

        if use_lora:
            if "FLUX" in mode or "Stable Diffusion 3" in mode:
                pipeline_configs["joint_attention_kwargs"] = {
                    "scale": lora_scale}
            else:
                pipeline_configs["cross_attention_kwargs"] = {
                    "scale": lora_scale}

        # Generate images
        images = pipeline(**pipeline_configs).images
    except Exception as e:
        raise gr.Error(f"Exception: {e}", duration=5)

    progress(100, "Completed!")
    del pipeline
    pipeline = None
    gc.collect()
    torch.cuda.empty_cache()
    return images


# -------------------------------------------- Gradio App -------------------------------------------- #
with gr.Blocks(title="TonAI Creative",
               theme=sty.app_theme,
               css=sty.custom_css) as interface:
    gr.HTML(sty.tonai_creative_html)
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("Basic Usage", open=True):
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate")
                with gr.Row():
                    width = gr.components.Slider(
                        minimum=512, maximum=1920, value=1024, step=8,
                        label="Width",
                        scale=1
                    )
                    height = gr.components.Slider(
                        minimum=512, maximum=1920, value=1024, step=8,
                        label="Height",
                        scale=1
                    )
                    mode = gr.Dropdown(
                        choices=TEXT_TO_IMAGE_DICTIONARY.keys(),
                        label="Mode",
                        filterable=False,
                        value=list(TEXT_TO_IMAGE_DICTIONARY.keys())[
                            0],  # FLUX.1 Merged is default
                        interactive=True,
                        scale=1)
                with gr.Row():
                    generate_btn = gr.Button("Generate", scale=2)
                    stop_btn = gr.Button("Stop", elem_id="stop-button", scale=1)

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="ugly, disfigured, deformed",
                    placeholder="Instruct the AI model that it should not include")
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            num_steps = gr.components.Slider(
                                minimum=3, maximum=50, value=20, step=1,
                                label="Inference Steps",
                                scale=2
                            )
                        with gr.Row():
                            guidance_scale = gr.components.Slider(
                                minimum=0, maximum=20, value=3, step=0.1,
                                label="CFG Scale",
                                scale=1
                            )
                        with gr.Row():
                            num_images = gr.components.Slider(
                                minimum=1, maximum=6, value=1, step=1,
                                label="Number of generated images",
                                scale=1)
                            scheduler = gr.Dropdown(
                                choices=SCHEDULERS,
                                label="Sampler",
                                filterable=False,
                                value=SCHEDULERS[0],
                                interactive=True,
                                scale=1)
                    with gr.Column(scale=1):
                        seed = gr.Textbox(label="RNG Seed", value=0)
                        rng_btn = gr.Button("Roll the 🎲", scale=1)
                        rng_btn.click(
                            fn=generate_number, inputs=None, outputs=seed)
                        fast_infer = gr.Checkbox(
                            label="Fast Inference",
                            info="Faster run with FP16",
                            value=True,
                            scale=1)
                with gr.Row():
                    lora_weight_file = gr.File(
                        label="LoRA safetensors file",
                        elem_classes="file-uploader",
                        file_types=["safetensors"],
                        min_width=50, height=30, scale=2)
                    lora_scale = gr.components.Slider(
                        minimum=0, maximum=1, value=0.8, step=0.01,
                        label="LoRA Scale",
                        scale=1
                    )
            with gr.Accordion("Helps", open=False):
                gr.Markdown(sty.tips_content)

        with gr.Column(scale=1):
            gallery = gr.Gallery(
                label="Generated Images",
                format="png",
                elem_id="gallery",
                columns=2, rows=2,
                preview=True,
                object_fit="contain")
            click_button_behavior = {
                "fn": gen_image,
                "outputs": gallery,
                "concurrency_limit": 10
            }
            click_event = generate_btn.click(inputs=[prompt,
                                       negative_prompt,
                                       width,
                                       height,
                                       num_steps,
                                       mode,
                                       seed,
                                       guidance_scale,
                                       lora_weight_file,
                                       lora_scale,
                                       fast_infer,
                                       scheduler,
                                       num_images],
                               **click_button_behavior)
            stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event])

        interface.load(
            lambda: gr.update(
                value=random.randint(
                    0, 999999)), None, seed)

if __name__ == '__main__':
    allowed_paths = ["stuffs/splash.png", "stuffs/favicon.png"]
    interface.queue(default_concurrency_limit=10)
    interface.launch(share=False,
                     root_path="/tonai",
                     server_name="0.0.0.0",
                     show_error=True,
                     favicon_path="stuffs/favicon.png",
                     allowed_paths=allowed_paths,
                     max_threads=10)
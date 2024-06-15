import os
import random
import torch
import gc
import gradio as gr
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler,\
                      StableDiffusionXLPipeline, StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from utils import *

def update_chat(entry_message):
    return gr.update(value=str(entry_message))

def back_to_home():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def activate_text2image():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def activate_text2text():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
def gen_image(prompt, negative_prompt, width, height, num_steps, mode, seed, guidance_scale, device):
    """
    Run diffusion model to generate image
    """
    device = f"cuda:{device.split('GPU')[1][1]}"
    guidance_scale = float(guidance_scale)
    generator = torch.Generator(device).manual_seed(int(seed))
    model_path = DIFFUSION_CHECKPOINTS[mode]["path"]
    Text2Image_class = globals()[DIFFUSION_CHECKPOINTS[mode]["pipeline"]]
    Text2Image_class.safety_checker=None
    if DIFFUSION_CHECKPOINTS[mode]["type"] == "pretrained":
        if DIFFUSION_CHECKPOINTS[mode]["pipeline"] == "StableDiffusion3Pipeline": # half precision for fp16
            pipeline = Text2Image_class.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            pipeline = Text2Image_class.from_pretrained(model_path)
    else:
        pipeline = Text2Image_class.from_single_file(model_path)

    if "lora" in DIFFUSION_CHECKPOINTS[mode]:
        directory, filename = os.path.split(DIFFUSION_CHECKPOINTS[mode]["lora"])
        pipeline.load_lora_weights(directory, weight_name=filename)
    if DIFFUSION_CHECKPOINTS[mode]["pipeline"] == "StableDiffusion3Pipeline":
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    try:
        pipeline = pipeline.to(device)
        image = pipeline(prompt=prompt,
                         negative_prompt=negative_prompt,
                         width=nearest_divisible_by_8(int(width)),
                         height=nearest_divisible_by_8(int(height)),
                         num_inference_steps=int(num_steps),
                         generator=generator,
                         guidance_scale=guidance_scale).images[0]
    except Exception as e:
        image = Image.open("stuffs/serverdown.jpg")
        print(e)
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    return image

with gr.Blocks(title="TonAI Space", theme=APP_THEME) as interface:
    with gr.Column(visible=False) as text2image:
        gr.HTML(tonai_creative_html)
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Instruct the AI model that it should not include")
                with gr.Row():
                    width = gr.Textbox(label="Image Width", value=768)
                    height = gr.Textbox(label="Image Height", value=768)
                with gr.Row():
                    seed = gr.Textbox(label="RNG Seed", value=0, scale=1)
                    guidance_scale = gr.Textbox(label="CFG Scale", value=7, scale=1)
                with gr.Row():
                    num_steps = gr.components.Slider(
                                    minimum=5, maximum=60, value=20, step=1,
                                    label="Inference Steps"
                                    )
                    mode=gr.Dropdown(choices=DIFFUSION_CHECKPOINTS.keys(), label="Mode",
                                     value=list(DIFFUSION_CHECKPOINTS.keys())[0])
                device_choices = display_gpu_info()
                device=gr.Dropdown(choices=device_choices, label="Device", value=device_choices[0])
                with gr.Row():
                    generate_btn = gr.Button("Generate")
                    back_to_home_btn_creative = gr.Button("🏠")
            with gr.Column(scale=2):
                generate_btn.click(
                            fn=gen_image,
                            inputs=[prompt, negative_prompt, width, height, num_steps, mode, seed, guidance_scale, device],
                            outputs=gr.Image(label="Generated Image", format="png"),
                            concurrency_limit=10
                        )

    with gr.Column(visible=False) as text2text:
        gr.HTML(tonai_chat_html)
        with gr.Row():
            chat_box = gr.TextArea(show_label=False)
        with gr.Row():
            text2text_input_text = gr.Textbox(show_label=False, placeholder="Message TonAI Lạc Đà")
        with gr.Row():
            send_btn = gr.Button("Send")
            back_to_home_btn_chat = gr.Button("🏠")
        send_btn.click(update_chat, inputs=[text2text_input_text], outputs=[chat_box])

    with gr.Column(visible=True) as homepage:
        gr.HTML(home_header_html)
        with gr.Row():
            text2image_btn = gr.Button("TonAI Creative")
            text2text_btn = gr.Button("TonAI Text Summarizer")

    back_to_home_btn_creative.click(back_to_home, inputs=[], outputs=[homepage, text2image, text2text])
    back_to_home_btn_chat.click(back_to_home, inputs=[], outputs=[homepage, text2image, text2text])
    text2image_btn.click(activate_text2image, inputs=[], outputs=[homepage, text2image, text2text])
    text2text_btn.click(activate_text2text, inputs=[], outputs=[homepage, text2image, text2text])
    interface.load(lambda: gr.update(value=random.randint(0, 999999)), None, seed)
    interface.load(lambda: gr.update(choices=display_gpu_info(), value=display_gpu_info()[0]), None, device)

if __name__ == '__main__':
    allowed_paths=["stuffs/tonai_research_logo.png"]
    interface.queue(default_concurrency_limit=10)
    interface.launch(share=False, allowed_paths=allowed_paths, max_threads=10)
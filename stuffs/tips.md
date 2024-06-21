# TonAI Creative Quick Start
### Basic Usage:
- **Prompt**: Enter the text prompt to generate your desired output. It should be less than 77 words.
- **Image Width** (and **Height**): Specify the width (and height) of the generated image.
- **Mode**: Choose the style of image you want to generate.
- For SD 1.5 models, optimized sizes (w x h) are 768x768 (square), 512x864 (portrait), 864x512 (landscape). For SDXL and SD 3 models, optimized sizes (w x h) are 1024x1024 (square), 768x1152 (portrait), 1152x768 (landscape). You can read the [Resolution cheatsheat](https://www.reddit.com/r/StableDiffusion/comments/15c3rf6/sdxl_resolution_cheat_sheet/) to obtain more information.

### Advanced Usage:
- **Negative Prompt**: Text prompts that instruct the AI model that it should not include certain elements in its generated images.
- **CFG Scale**: A parameter that controls how much the image generation process follows the text prompt. The higher the value, the more the image sticks to a given text input.
- **Inference Step**: The number of steps the model takes to generate the output.
- **LoRA safetensor File**: Upload a Low-Rank Adaptation (LoRA) safetensor file to fine-tune the model. You can seek the LoRA weight on the Internet, and add the tag with syntax `<lora:[scale]>` to your prompt. For example, with LoRA file `Mod1_blah.safetensor`, you can indicate this object in prompt by calling `Mod1_blah` (your file name without extension) and tag `<lora:0.66>` (0.66 is the scale of adapter, default value is 1).

### More Information:
Except Stable Diffusion 3 Medium model, **TonAI Creative** uses `DPM++ 2M SDE Karras` Sampler for all pipelines. We value your feedback! Please share your thoughts and suggestions with us at tungnguyen99.tn@gmail.com
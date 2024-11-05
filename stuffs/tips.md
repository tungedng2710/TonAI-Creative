# Quick Start
- **Prompt**: Text prompt to generate your desired output.
- **Width** (and **Height**): Specify the width (and height) of the generated image.
- **Fast Inference**: Model will be quantized to half-precision to boost the inference speed.
- **Negative Prompt**: Text prompts that instruct the AI model that it should not include certain elements in its generated images. (Not available for FLUX Pipeline)
- **CFG Scale**: A parameter that controls how much the image generation process follows the text prompt. The higher the value, the more the image sticks to a given text input.
- **Inference Steps**: The number of steps the model takes to generate the output.
- **Mode**: Choose the engine to generate images.
- **LoRA safetensor File**: Upload a Low-Rank Adaptation (LoRA) ```safetensors``` file to fine-tune the model.
- **Sampler**: Method of generating data in a specific way. Set its value as ```Default``` to use the default config of pipeline
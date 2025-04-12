import torch
from PIL import Image
from diffusers import DiffusionPipeline

import gc
import os

# Optional: Helps with CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@torch.no_grad()
def generate_background_image(
    salient_img: Image.Image,
    bg_mask: Image.Image,
    prompt: str,
    seed: int = 990099,
    num_steps: int = 50,
    cond_scale: float = 1.0,
    reproducibility: bool = True
):
    # Ensure image types are correct
    if not isinstance(salient_img, Image.Image):
        salient_img = Image.fromarray(salient_img)
    if not isinstance(bg_mask, Image.Image):
        bg_mask = Image.fromarray(bg_mask)

    model_id = "yahoo-inc/photo-background-generation"
    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline=model_id
    ).to("cuda")


    # Set reproducible seed if needed
    generator = (
        torch.Generator(device='cuda').manual_seed(seed)
        if reproducibility else None
    )

    # Run inference using autocast for mixed precision
    with torch.autocast("cuda", dtype=torch.float32):
        result = pipeline(
            prompt=prompt,
            image=salient_img,
            mask_image=bg_mask,
            control_image=bg_mask,
            num_images_per_prompt=1,
            generator=generator,
            num_inference_steps=num_steps,
            guess_mode=False,
            controlnet_conditioning_scale=cond_scale
        ).images[0]

    # Clean up GPU memory immediately
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

    return result

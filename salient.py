# SALIENT

import os
import numpy as np
from PIL import Image, ImageOps
from diffusers.utils import load_image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer

from model.evf_sam2 import EvfSam2Model
from model.segment_anything.utils.transforms import ResizeLongestSide

def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    '''
    preprocess of Segment Anything Model, including scaling, normalization and padding.  
    preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
    input: ndarray
    output: torch.Tensor
    '''
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    
    # Normalize colors
    if model_type=="ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = (x - pixel_mean) / pixel_std
        # Pad
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    else:
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
        x = (x - pixel_mean) / pixel_std
        resize_shape = None
    
    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)


@torch.no_grad()
def pred(image_np, prompt, semantic_type, model_type="sam2"):

    version = "YxZhang/evf-sam2-multitask"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(version, padding_side="right", use_fast=False)

    kwargs = { "torch_dtype": torch.half }
    if model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(version, low_cpu_mem_usage=True, **kwargs).cuda().eval()
    elif model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(version, low_cpu_mem_usage=True, **kwargs).cuda().eval()
    elif model_type=="sam2":
        from model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(version, low_cpu_mem_usage=True, **kwargs)
        del model.visual_model.memory_encoder
        del model.visual_model.memory_attention
        model = model.eval()
        model.to(device)


    original_size_list = [image_np.shape[:2]]

    image_beit = beit3_preprocess(image_np, 224).to(dtype=model.dtype, device=model.device)
    image_sam, resize_shape = sam_preprocess(image_np, model_type="sam2")
    image_sam = image_sam.to(dtype=model.dtype, device=model.device)

    if semantic_type:
        prompt = "[semantic] " + prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    # inference
    pred_mask = model.inference(
        image_sam.unsqueeze(0),
        image_beit.unsqueeze(0),
        input_ids,
        resize_list=[resize_shape],
        original_size_list=original_size_list,
    )
    pred_mask = pred_mask.detach().cpu().numpy()[0]
    pred_mask = pred_mask > 0

    # visualization
    visualization = image_np.copy()
    visualization[pred_mask] = (
        image_np * 0.5
        + pred_mask[:, :, None].astype(np.uint8) * np.array([220, 120, 50]) * 0.5
    )[pred_mask]

    return visualization / 255.0, pred_mask.astype(np.float16)

def resize_with_padding(img: Image.Image, expected_size=(512, 512)) -> Image.Image:
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def extract_salient_object(
    image_path: str,
    prompt: str,
    semantic_type: bool = False,
    salient_save_path: str = None,
    mask_save_path: str = None,
    background_mask_save_path: str = None,
    return_salient_mask: bool = False,
    WIDTH: int = 512,
    HEIGHT: int = 512
):
    # Load and preprocess image
    image_pil = load_image(image_path).convert("RGB")
    image_pil = resize_with_padding(image_pil, (WIDTH, HEIGHT))
    image_np = np.array(image_pil)

    # Predict mask using provided method
    vis_np, mask_np = pred(image_np, prompt, semantic_type=semantic_type)

    # Ensure binary mask
    mask = mask_np.astype(bool)

    # Create alpha channel from mask
    alpha = np.zeros_like(mask_np, dtype=np.uint8)
    alpha[mask] = 255
    rgba = np.dstack((image_np, alpha))
    result_image = Image.fromarray(rgba, mode='RGBA')

    # Save the transparent foreground image
    if salient_save_path:
        os.makedirs(os.path.dirname(salient_save_path), exist_ok=True)
        result_image.save(salient_save_path)

    # Save the foreground mask
    if mask_save_path:
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img.save(mask_save_path)

    # Save the background mask
    background_mask_np = (~mask).astype(np.uint8)
    background_mask_img = Image.fromarray(background_mask_np * 255, mode='L')
    if background_mask_save_path:
        os.makedirs(os.path.dirname(background_mask_save_path), exist_ok=True)
        background_mask_img.save(background_mask_save_path)

    # Return output
    if return_salient_mask:
        mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255, mode='L')
        return result_image, mask_pil
    return result_image, background_mask_img

"""
CLIP Score: Measures how well a generated image matches a text prompt by computing cosine similarity between their CLIP embeddings.  
Object Similarity: Evaluates if the main object's identity is preserved after background generation using cosine similarity of BLIP-2 image embeddings.  
Object Expansion: Quantifies how much the object has grown in pixel area after outpainting compared to the original object-only image.
"""

import torch
import numpy as np
from typing import Union
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForImageTextRetrieval,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_clip_score(text_prompt: str, image_np: np.ndarray) -> float:
    """Computes CLIP score between a text prompt and image.

    Args:
        text_prompt: Input text prompt.
        image_np: Numpy array of image in RGB format.

    Returns:
        Cosine similarity score between CLIP text and image embeddings.
    """
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image = Image.fromarray(image_np.astype(np.uint8))
    inputs = clip_processor(text=[text_prompt], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        text_emb = outputs.text_embeds.detach().cpu().numpy()
        image_emb = outputs.image_embeds.detach().cpu().numpy()

    return cosine_similarity(text_emb, image_emb)[0][0]


def compute_blip_similarity(
    image_a_np: np.ndarray, image_b_np: np.ndarray
) -> float:
    """Computes object similarity using BLIP embeddings.

    Args:
        image_a_np: First image (e.g., object-only) in RGB format.
        image_b_np: Second image (e.g., outpainted) in RGB format.

    Returns:
        Cosine similarity between image embeddings from BLIP.
    """
    blip_model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco"
    ).to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

    def get_embedding(image_np: np.ndarray) -> np.ndarray:
        image = Image.fromarray(image_np.astype(np.uint8))
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = blip_model.vision_model(**inputs)
            pooled = outputs.pooler_output
        return pooled.cpu().numpy()

    emb_a = get_embedding(image_a_np)
    emb_b = get_embedding(image_b_np)
    return cosine_similarity(emb_a, emb_b)[0][0]


def compute_object_expansion(
    image_object_np: np.ndarray, image_outpainted_np: np.ndarray
) -> Union[float, None]:
    """Computes object expansion as ratio of pixel area.

    Args:
        image_object_np: Object-only image (RGB).
        image_outpainted_np: Outpainted image (RGB).

    Returns:
        Expansion ratio (float) or None if denominator is zero.
    """
    def get_area(img: np.ndarray) -> int:
        gray = np.mean(img, axis=-1)
        mask = gray > 30  # Simple brightness threshold
        return np.sum(mask)

    area_object = get_area(image_object_np)
    area_outpainted = get_area(image_outpainted_np)

    if area_object == 0:
        return None
    return area_outpainted / area_object

def compute_ciou(
    image_object_np: np.ndarray, image_outpainted_np: np.ndarray
) -> Union[float, None]:
    """Computes cumulative Intersection over Union (cIoU) between two binary masks.

    Args:
        image_object_np: Object-only image (RGB).
        image_outpainted_np: Outpainted image (RGB).

    Returns:
        cIoU value (float) or None if union is zero.
    """
    def get_binary_mask(img: np.ndarray) -> np.ndarray:
        gray = np.mean(img, axis=-1)
        return (gray > 30).astype(np.uint8)

    mask_object = get_binary_mask(image_object_np)
    mask_outpainted = get_binary_mask(image_outpainted_np)

    intersection = np.logical_and(mask_object, mask_outpainted).sum()
    union = np.logical_or(mask_object, mask_outpainted).sum()

    if union == 0:
        return None
    return intersection / union

def main(
    text_prompt: str,
    object_only_img_np: np.ndarray,
    outpainted_img_np: np.ndarray,
) -> None:
    """Computes and prints CLIP score, object similarity, expansion ratio, and cIoU."""
    clip_score = compute_clip_score(text_prompt, outpainted_img_np)
    object_similarity = compute_blip_similarity(object_only_img_np, outpainted_img_np)
    object_expansion = compute_object_expansion(object_only_img_np, outpainted_img_np)
    ciou_score = compute_ciou(object_only_img_np, outpainted_img_np)

    print(f"CLIP Score (text-image alignment): {clip_score:.4f}")
    print(f"Object Similarity (BLIP cosine): {object_similarity:.4f}")
    print(f"Object Expansion (area ratio): {object_expansion:.4f}")
    print(f"cIoU (mask intersection-over-union): {ciou_score:.4f}")

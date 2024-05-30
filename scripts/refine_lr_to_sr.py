import torch
import os

import numpy as np
from hashlib import md5
def hash_img(img):
    return md5(np.array(img).tobytes()).hexdigest()
def hash_any(obj):
    return md5(str(obj).encode()).hexdigest()

def refine_lr_with_sd(pil_image_list, concept_img_list, control_image_list, prompt_list, pipe=None, strength=0.35, neg_prompt_list="", output_size=(512, 512), controlnet_conditioning_scale=1.):
    with torch.no_grad():
        images = pipe(
            image=pil_image_list,
            ip_adapter_image=concept_img_list,
            prompt=prompt_list,
            neg_prompt=neg_prompt_list,
            num_inference_steps=50,
            strength=strength,
            height=output_size[0],
            width=output_size[1],
            control_image=control_image_list,
            guidance_scale=5.0,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.manual_seed(233),
        ).images
    return images

SR_cache = None

def run_sr_fast(source_pils, scale=4):
    from PIL import Image
    from scripts.upsampler import RealESRGANer
    import numpy as np
    global SR_cache
    if SR_cache is not None:
        upsampler = SR_cache
    else:
        upsampler = RealESRGANer(
            scale=4,
            onnx_path="ckpt/realesrgan-x4.onnx",
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0,
        )
    ret_pils = []
    for idx, img_pils in enumerate(source_pils):
        np_in = isinstance(img_pils, np.ndarray)
        assert isinstance(img_pils, (Image.Image, np.ndarray))
        img = np.array(img_pils)
        output, _ = upsampler.enhance(img, outscale=scale)
        if np_in:
            ret_pils.append(output)
        else:
            ret_pils.append(Image.fromarray(output))
    if SR_cache is None:
        SR_cache = upsampler
    return ret_pils

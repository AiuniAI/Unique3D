from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
import torch
from copy import deepcopy

ENABLE_CPU_CACHE = False
DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"

cached_models = {}  # cache for models to avoid repeated loading, key is model name
def cache_model(func):
    def wrapper(*args, **kwargs):
        if ENABLE_CPU_CACHE:
            model_name = func.__name__ + str(args) + str(kwargs)
            if model_name not in cached_models:
                cached_models[model_name] = func(*args, **kwargs)
            return cached_models[model_name]
        else:
            return func(*args, **kwargs)
    return wrapper

def copied_cache_model(func):
    def wrapper(*args, **kwargs):
        if ENABLE_CPU_CACHE:
            model_name = func.__name__ + str(args) + str(kwargs)
            if model_name not in cached_models:
                cached_models[model_name] = func(*args, **kwargs)
            return deepcopy(cached_models[model_name])
        else:
            return func(*args, **kwargs)
    return wrapper

def model_from_ckpt_or_pretrained(ckpt_or_pretrained, model_cls, original_config_file='ckpt/v1-inference.yaml', torch_dtype=torch.float16, **kwargs):
    if ckpt_or_pretrained.endswith(".safetensors"):
        pipe = model_cls.from_single_file(ckpt_or_pretrained, original_config_file=original_config_file, torch_dtype=torch_dtype, **kwargs)
    else:
        pipe = model_cls.from_pretrained(ckpt_or_pretrained, torch_dtype=torch_dtype, **kwargs)
    return pipe

@copied_cache_model
def load_base_model_components(base_model=DEFAULT_BASE_MODEL, torch_dtype=torch.float16):
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        requires_safety_checker=False, 
        safety_checker=None,
    )
    pipe: StableDiffusionPipeline = model_from_ckpt_or_pretrained(
        base_model,
        StableDiffusionPipeline,
        **model_kwargs
    )
    pipe.to("cpu")
    return pipe.components

@cache_model
def load_controlnet(controlnet_path, torch_dtype=torch.float16):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)
    return controlnet

@cache_model
def load_image_encoder():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
    return image_encoder

def load_common_sd15_pipe(base_model=DEFAULT_BASE_MODEL, device="auto", controlnet=None, ip_adapter=False, plus_model=True, torch_dtype=torch.float16, model_cpu_offload_seq=None, enable_sequential_cpu_offload=False, vae_slicing=False, pipeline_class=None, **kwargs):
    model_kwargs = dict(
        torch_dtype=torch_dtype, 
        device_map=device,
        requires_safety_checker=False, 
        safety_checker=None,
    )
    components = load_base_model_components(base_model=base_model, torch_dtype=torch_dtype)
    model_kwargs.update(components)
    model_kwargs.update(kwargs)
    
    if controlnet is not None:
        if isinstance(controlnet, list):
            controlnet = [load_controlnet(controlnet_path, torch_dtype=torch_dtype) for controlnet_path in controlnet]
        else:
            controlnet = load_controlnet(controlnet, torch_dtype=torch_dtype)
        model_kwargs.update(controlnet=controlnet)
    
    if pipeline_class is None:
        if controlnet is not None:
            pipeline_class = StableDiffusionControlNetPipeline
        else:
            pipeline_class = StableDiffusionPipeline
    
    pipe: StableDiffusionPipeline = model_from_ckpt_or_pretrained(
        base_model,
        pipeline_class,
        **model_kwargs
    )

    if ip_adapter:
        image_encoder = load_image_encoder()
        pipe.image_encoder = image_encoder
        if plus_model:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")
        else:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
        pipe.set_ip_adapter_scale(1.0)
    else:
        pipe.unload_ip_adapter()
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if model_cpu_offload_seq is None:
        if isinstance(pipe, StableDiffusionControlNetPipeline):
            pipe.model_cpu_offload_seq = "text_encoder->controlnet->unet->vae"
        elif isinstance(pipe, StableDiffusionControlNetImg2ImgPipeline):
            pipe.model_cpu_offload_seq = "text_encoder->controlnet->vae->unet->vae"
    else:
        pipe.model_cpu_offload_seq = model_cpu_offload_seq
    
    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to("cuda")
        pass
        # pipe.enable_model_cpu_offload()
    if vae_slicing:
        pipe.enable_vae_slicing()
        
    import gc
    gc.collect()
    return pipe


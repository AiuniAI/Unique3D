import torch
from typing import List
from dataclasses import dataclass
from app.utils import rgba_to_rgb
from custum_3d_diffusion.trainings.config_classes import ExprimentConfig, TrainerSubConfig
from custum_3d_diffusion import modules
from custum_3d_diffusion.custum_modules.unifield_processor import AttnConfig, ConfigurableUNet2DConditionModel
from custum_3d_diffusion.trainings.base import BasicTrainer
from custum_3d_diffusion.trainings.utils import load_config


@dataclass
class FakeAccelerator:
    device: torch.device = torch.device("cuda")


def init_trainers(cfg_path: str, weight_dtype: torch.dtype, extras: dict):
    accelerator = FakeAccelerator()
    cfg: ExprimentConfig = load_config(ExprimentConfig, cfg_path, extras)
    init_config: AttnConfig = load_config(AttnConfig, cfg.init_config)
    configurable_unet = ConfigurableUNet2DConditionModel(init_config, weight_dtype)
    configurable_unet.enable_xformers_memory_efficient_attention()
    trainer_cfgs: List[TrainerSubConfig] = [load_config(TrainerSubConfig, trainer) for trainer in cfg.trainers]
    trainers: List[BasicTrainer] = [modules.find(trainer.trainer_type)(accelerator, None, configurable_unet, trainer.trainer, weight_dtype, i) for i, trainer in enumerate(trainer_cfgs)]
    return trainers, configurable_unet

from app.utils import make_image_grid, split_image
def process_image(function, img, guidance_scale=2., merged_image=False, remove_bg=True):
    from rembg import remove
    if remove_bg:
        img = remove(img)
    img = rgba_to_rgb(img)
    if merged_image:
        img = split_image(img, rows=2)
    images = function(
        image=img,
        guidance_scale=guidance_scale,
    )
    if len(images) > 1:
        return make_image_grid(images, rows=2)
    else:
        return images[0]


def process_text(trainer, pipeline, img, guidance_scale=2.):
    pipeline.cfg.validation_prompts = [img]
    titles, images = trainer.batched_validation_forward(pipeline, guidance_scale=[guidance_scale])
    return images[0]


def load_pipeline(config_path, ckpt_path, pipeline_filter=lambda x: True, weight_dtype = torch.bfloat16):
    training_config = config_path
    load_from_checkpoint = ckpt_path
    extras = []
    device = "cuda"
    trainers, configurable_unet = init_trainers(training_config, weight_dtype, extras)
    shared_modules = dict()
    for trainer in trainers:
        shared_modules = trainer.init_shared_modules(shared_modules)

    if load_from_checkpoint is not None:
        state_dict = torch.load(load_from_checkpoint)
        configurable_unet.unet.load_state_dict(state_dict, strict=False)
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    configurable_unet.unet.to(device, dtype=weight_dtype)

    pipeline = None
    trainer_out = None
    for trainer in trainers:
        if pipeline_filter(trainer.cfg.trainer_name):
            pipeline = trainer.construct_pipeline(shared_modules, configurable_unet.unet)
            pipeline.set_progress_bar_config(disable=False)
            trainer_out = trainer
    pipeline = pipeline.to(device)
    return trainer_out, pipeline
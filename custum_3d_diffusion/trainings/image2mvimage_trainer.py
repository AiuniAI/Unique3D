import torch
from diffusers import AutoencoderKL, DDPMScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, BatchFeature

import json
from dataclasses import dataclass
from typing import List, Optional

from custum_3d_diffusion.modules import register
from custum_3d_diffusion.trainings.base import BasicTrainer
from custum_3d_diffusion.custum_pipeline.unifield_pipeline_img2mvimg import StableDiffusionImage2MVCustomPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

def get_HW(resolution):
    if isinstance(resolution, str):
        resolution = json.loads(resolution)
    if isinstance(resolution, int):
        H = W = resolution
    elif isinstance(resolution, list):
        H, W = resolution
    return H, W

@register("image2mvimage_trainer")
class Image2MVImageTrainer(BasicTrainer):
    """
    Trainer for simple image to multiview images.
    """
    @dataclass
    class TrainerConfig(BasicTrainer.TrainerConfig):
        trainer_name: str = "image2mvimage"
        condition_image_column_name: str = "conditioning_image"
        image_column_name: str = "image"
        condition_dropout: float = 0.
        condition_image_resolution: str = "512"
        validation_images: Optional[List[str]] = None
        noise_offset: float = 0.1                           
        max_loss_drop: float = 0.                           
        snr_gamma: float = 5.0                              
        log_distribution: bool = False
        latents_offset: Optional[List[float]] = None
        input_perturbation: float = 0.
        noisy_condition_input: bool = False                 # whether to add noise for ref unet input
        normal_cls_offset: int = 0
        condition_offset: bool = True
        zero_snr: bool = False
        linear_beta_schedule: bool = False

    cfg: TrainerConfig

    def configure(self) -> None:
        return super().configure()

    def init_shared_modules(self, shared_modules: dict) -> dict:
        if 'vae' not in shared_modules:
            vae = AutoencoderKL.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.weight_dtype
            )
            vae.requires_grad_(False)
            vae.to(self.accelerator.device, dtype=self.weight_dtype)
            shared_modules['vae'] = vae
        if 'image_encoder' not in shared_modules:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="image_encoder"
            )
            image_encoder.requires_grad_(False)
            image_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
            shared_modules['image_encoder'] = image_encoder
        if 'feature_extractor' not in shared_modules:
            feature_extractor = CLIPImageProcessor.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="feature_extractor"
            )
            shared_modules['feature_extractor'] = feature_extractor
        return shared_modules

    def init_train_dataloader(self, shared_modules: dict) -> torch.utils.data.DataLoader:
        raise NotImplementedError()

    def loss_rescale(self, loss, timesteps=None):
        raise NotImplementedError()

    def forward_step(self, batch, unet, shared_modules, noise_scheduler: DDPMScheduler, global_step) -> torch.Tensor:
        raise NotImplementedError()
    
    def construct_pipeline(self, shared_modules, unet, old_version=False):
        MyPipeline = StableDiffusionImage2MVCustomPipeline
        pipeline = MyPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            vae=shared_modules['vae'],
            image_encoder=shared_modules['image_encoder'],
            feature_extractor=shared_modules['feature_extractor'],
            unet=unet,
            safety_checker=None,
            torch_dtype=self.weight_dtype,
            latents_offset=self.cfg.latents_offset,
            noisy_cond_latents=self.cfg.noisy_condition_input,
            condition_offset=self.cfg.condition_offset,
        )
        pipeline.set_progress_bar_config(disable=True)
        scheduler_dict = {}
        if self.cfg.zero_snr:
            scheduler_dict.update(rescale_betas_zero_snr=True)
        if self.cfg.linear_beta_schedule:
            scheduler_dict.update(beta_schedule='linear')
        
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_dict)
        return pipeline

    def get_forward_args(self):
        if self.cfg.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(self.cfg.seed)
        
        H, W = get_HW(self.cfg.resolution)
        H_cond, W_cond = get_HW(self.cfg.condition_image_resolution)

        sub_img_H = H // 2
        num_imgs = H // sub_img_H * W // sub_img_H

        forward_args = dict(
            num_images_per_prompt=num_imgs,
            num_inference_steps=50,
            height=sub_img_H,
            width=sub_img_H,
            height_cond=H_cond,
            width_cond=W_cond,
            generator=generator,
        )
        if self.cfg.zero_snr:
            forward_args.update(guidance_rescale=0.7)
        return forward_args

    def pipeline_forward(self, pipeline, **pipeline_call_kwargs) -> StableDiffusionPipelineOutput:
        forward_args = self.get_forward_args()
        forward_args.update(pipeline_call_kwargs)
        return pipeline(**forward_args)

    def batched_validation_forward(self, pipeline, **pipeline_call_kwargs) -> tuple:
        raise NotImplementedError()
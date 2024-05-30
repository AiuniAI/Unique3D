import json
import torch
from diffusers import EulerAncestralDiscreteScheduler, DDPMScheduler
from dataclasses import dataclass

from custum_3d_diffusion.modules import register
from custum_3d_diffusion.trainings.image2mvimage_trainer import Image2MVImageTrainer
from custum_3d_diffusion.custum_pipeline.unifield_pipeline_img2img import StableDiffusionImageCustomPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

def get_HW(resolution):
    if isinstance(resolution, str):
        resolution = json.loads(resolution)
    if isinstance(resolution, int):
        H = W = resolution
    elif isinstance(resolution, list):
        H, W = resolution
    return H, W


@register("image2image_trainer")
class Image2ImageTrainer(Image2MVImageTrainer):
    """
    Trainer for simple image to multiview images.
    """
    @dataclass
    class TrainerConfig(Image2MVImageTrainer.TrainerConfig):
        trainer_name: str = "image2image"

    cfg: TrainerConfig

    def forward_step(self, batch, unet, shared_modules, noise_scheduler: DDPMScheduler, global_step) -> torch.Tensor:
        raise NotImplementedError()

    def construct_pipeline(self, shared_modules, unet, old_version=False):
        MyPipeline = StableDiffusionImageCustomPipeline
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

        forward_args = dict(
            num_images_per_prompt=1,
            num_inference_steps=20,
            height=H,
            width=W,
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
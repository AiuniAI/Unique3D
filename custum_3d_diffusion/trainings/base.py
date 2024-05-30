import torch
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from dataclasses import dataclass, field
from typing import Optional, Union
from datasets import load_dataset
import json
import abc
from diffusers.utils import make_image_grid
import numpy as np
import wandb

from custum_3d_diffusion.trainings.utils import load_config
from custum_3d_diffusion.custum_modules.unifield_processor import ConfigurableUNet2DConditionModel, AttnConfig

class BasicTrainer(torch.nn.Module, abc.ABC):
    accelerator: Accelerator
    logger: MultiProcessAdapter
    unet: ConfigurableUNet2DConditionModel
    train_dataloader: torch.utils.data.DataLoader
    test_dataset: torch.utils.data.Dataset
    attn_config: AttnConfig
    
    @dataclass
    class TrainerConfig:
        trainer_name: str = "basic"
        pretrained_model_name_or_path: str = ""
        
        attn_config: dict = field(default_factory=dict)
        dataset_name: str = ""
        dataset_config_name: Optional[str] = None
        resolution: str = "1024"
        dataloader_num_workers: int = 4
        pair_sampler_group_size: int = 1
        num_views: int = 4
        
        max_train_steps: int = -1                       # -1 means infinity, otherwise [0, max_train_steps)
        training_step_interval: int = 1                 # train on step i*interval, stop at max_train_steps
        max_train_samples: Optional[int] = None
        seed: Optional[int] = None                      # For dataset related operations and validation stuff
        train_batch_size: int = 1
        
        validation_interval: int = 5000
        debug: bool = False
    
    cfg: TrainerConfig    # only enable_xxx is used
    
    def __init__(
        self, 
        accelerator: Accelerator, 
        logger: MultiProcessAdapter,
        unet: ConfigurableUNet2DConditionModel,
        config: Union[dict, str],
        weight_dtype: torch.dtype,
        index: int,
    ):
        super().__init__()
        self.index = index              # index in all trainers
        self.accelerator = accelerator
        self.logger = logger
        self.unet = unet
        self.weight_dtype = weight_dtype
        self.ext_logs = {}
        self.cfg = load_config(self.TrainerConfig, config)
        self.attn_config = load_config(AttnConfig, self.cfg.attn_config)
        self.test_dataset = None
        self.validate_trainer_config()
        self.configure()
    
    def get_HW(self):
        resolution = json.loads(self.cfg.resolution)
        if isinstance(resolution, int):
            H = W = resolution
        elif isinstance(resolution, list):
            H, W = resolution
        return H, W
    
    def unet_update(self):
        self.unet.update_config(self.attn_config)
    
    def validate_trainer_config(self):
        pass
    
    def is_train_finished(self, current_step):
        assert isinstance(self.cfg.max_train_steps, int)
        return self.cfg.max_train_steps != -1 and current_step >= self.cfg.max_train_steps
    
    def next_train_step(self, current_step):
        if self.is_train_finished(current_step):
            return None
        return current_step + self.cfg.training_step_interval

    @classmethod
    def make_image_into_grid(cls, all_imgs, rows=2, columns=2):
        catted = [make_image_grid(all_imgs[i:i+rows * columns], rows=rows, cols=columns) for i in range(0, len(all_imgs), rows * columns)]
        return make_image_grid(catted, rows=1, cols=len(catted))

    def configure(self) -> None:
        pass
    
    @abc.abstractmethod
    def init_shared_modules(self, shared_modules: dict) -> dict:
        pass
    
    def load_dataset(self):
        dataset = load_dataset(
            self.cfg.dataset_name,
            self.cfg.dataset_config_name,
            trust_remote_code=True
        )
        return dataset

    @abc.abstractmethod
    def init_train_dataloader(self, shared_modules: dict) -> torch.utils.data.DataLoader:
        """Both init train_dataloader and test_dataset, but returns train_dataloader only"""
        pass
    
    @abc.abstractmethod
    def forward_step(
        self, 
        *args, 
        **kwargs
    ) -> torch.Tensor:
        """
        input a batch
        return a loss
        """
        self.unet_update()
        pass
    
    @abc.abstractmethod
    def construct_pipeline(self, shared_modules, unet):
        pass
    
    @abc.abstractmethod
    def pipeline_forward(self, pipeline, **pipeline_call_kwargs) -> tuple:
        """
            For inference time forward.
        """
        pass

    @abc.abstractmethod
    def batched_validation_forward(self, pipeline, **pipeline_call_kwargs) -> tuple:
        pass

    def do_validation(
        self,
        shared_modules,
        unet,
        global_step,
    ):
        self.unet_update()
        self.logger.info("Running validation... ")
        pipeline = self.construct_pipeline(shared_modules, unet)
        pipeline.set_progress_bar_config(disable=True)
        titles, images = self.batched_validation_forward(pipeline, guidance_scale=[1., 3.])
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
            elif tracker.name == "wandb":
                [image.thumbnail((512, 512)) for image, title in zip(images, titles) if 'noresize' not in title]   # inplace operation
                tracker.log({"validation": [
                    wandb.Image(image, caption=f"{i}: {titles[i]}", file_type="jpg")
                    for i, image in enumerate(images)]})
            else:
                self.logger.warn(f"image logging not implemented for {tracker.name}")
        del pipeline
        torch.cuda.empty_cache()
        return images

    
    @torch.no_grad()
    def log_validation(
        self,
        shared_modules,
        unet,
        global_step,
        force=False
    ):
        if self.accelerator.is_main_process:
            for tracker in self.accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(self.ext_logs)
        self.ext_logs = {}
        if (global_step % self.cfg.validation_interval == 0 and not self.is_train_finished(global_step)) or force:
            self.unet_update()
            if self.accelerator.is_main_process:
                self.do_validation(shared_modules, self.accelerator.unwrap_model(unet), global_step)

    def save_model(self, unwrap_unet, shared_modules, save_dir):
        if self.accelerator.is_main_process:
            pipeline = self.construct_pipeline(shared_modules, unwrap_unet)
            pipeline.save_pretrained(save_dir)
            self.logger.info(f"{self.cfg.trainer_name} Model saved at {save_dir}")

    def save_debug_info(self, save_name="debug", **kwargs):
        if self.cfg.debug:
            to_saves = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
            import pickle
            import os
            if os.path.exists(f"{save_name}.pkl"):
                for i in range(100):
                    if not os.path.exists(f"{save_name}_v{i}.pkl"):
                        save_name = f"{save_name}_v{i}"
                        break
            with open(f"{save_name}.pkl", "wb") as f:
                pickle.dump(to_saves, f)
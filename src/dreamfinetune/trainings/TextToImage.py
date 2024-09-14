import itertools
import math
import os

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, HfApi
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from ..utils import DreamBoothTextToImageDataset


class StableDiffusionTextToImageFineTune:

    def __init__(self, pretrained_model_name_or_path, **kwargs):
        """
        Ініціалізує об'єкт для fine-tuning моделі Stable Diffusion на завданні inpainting.

        Args:
            pretrained_model_name_or_path (str): Шлях або ім'я попередньо натренованої моделі.
            resolution (int): Роздільна здатність зображень, до якої будуть масштабуватися вхідні дані.
            center_crop (bool): Використовувати центральний crop під час підготовки даних.
            train_text_encoder (bool): Визначає, чи потрібно тренувати текстовий енкодер.
            dataset (str): Об'єкт датасету, який використовується для тренування.
            dataset_config_name (str): Назва конфігурації датасету, якщо є декілька варіантів.
            dataset_row_name (str): Назва рядка в датасеті, який використовується для тренування.
            instance_prompt (str): Текстовий prompt, що описує індивідуальні зображення.
            learning_rate (float): Початковий темп навчання.
            max_training_steps (int): Максимальна кількість кроків тренування.
            save_steps (int): Кількість кроків між збереженням контрольних точок.
            train_batch_size (int): Розмір батчу для тренування.
            gradient_accumulation_steps (int): Кількість кроків для акумуляції градієнтів перед оновленням параметрів.
            max_grad_norm (float): Максимальна норма градієнта для обрізки.
            mixed_precision (str): Використання змішаної точності ("fp16", "bf16" або None).
            gradient_checkpointing (bool): Використання збереження контрольних точок градієнта для зменшення використання пам'яті.
            use_8bit_adam (bool): Використання 8-бітного Adam оптимізатора для зменшення використання пам'яті.
            seed (int): Випадковий seed для повторюваності результатів.
            with_prior_preservation (bool): Використання збереження prior під час тренування для підтримки специфіки індивідуальних зображень.
            prior_loss_weight (float): Вага втрат prior у загальній функції втрат.
            sample_batch_size (int): Розмір батчу для генерації зразків під час тренування.
            class_data_dir (str): Директорія з класовими зображеннями для prior preservation.
            class_prompt (str): Текстовий prompt, що описує класові зображення.
            num_class_images (int): Кількість зображень для кожного класу.
            lr_scheduler (str): Планувальник темпу навчання ("linear", "cosine", тощо).
            lr_warmup_steps (int): Кількість кроків для прогріву темпу навчання.
            output_dir (str): Директорія для збереження результатів тренування.
            push_to_hub (bool): Чи завантажувати результати на Hugging Face Hub.
            repo_id (str): Ідентифікатор репозиторію на Hugging Face Hub.
            token (str): Токен для аутентифікації на Hugging Face Hub.
        """
        self.max_train_steps = kwargs.get('max_train_steps', 500)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.resolution = kwargs.get('resolution', 512)
        self.center_crop = kwargs.get('center_crop', False)
        self.train_text_encoder = kwargs.get('train_text_encoder', False)
        self.dataset = kwargs.get('dataset', None)
        self.dataset_config_name = kwargs.get('dataset_config_name', None)
        self.dataset_column = kwargs.get('dataset_column', None)
        self.instance_prompt = kwargs.get('instance_prompt', None)
        self.learning_rate = kwargs.get('learning_rate', 5e-06)
        self.save_steps = kwargs.get('save_steps', 250)
        self.train_batch_size = kwargs.get('train_batch_size', 2)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 2)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.mixed_precision = kwargs.get('mixed_precision', "fp16")
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        self.use_8bit_adam = kwargs.get('use_8bit_adam', True)
        self.seed = kwargs.get('seed', 3434554)
        self.with_prior_preservation = kwargs.get('with_prior_preservation', False)
        self.prior_loss_weight = kwargs.get('prior_loss_weight', None)
        self.sample_batch_size = kwargs.get('sample_batch_size', 2)
        self.class_data_dir = kwargs.get('class_data_dir', None)
        self.class_prompt = kwargs.get('class_prompt', None)
        self.num_class_images = kwargs.get('num_class_images', None)
        self.lr_scheduler = kwargs.get('lr_scheduler', 'constant')
        self.lr_warmup_steps = kwargs.get('lr_warmup_steps', 100)
        self.output_dir = kwargs.get('output_dir', 'dreambooth')
        self.push_to_hub = kwargs.get('push_to_hub', True)
        self.repo_id = kwargs.get('repo_id', None)
        self.token = kwargs.get('token', None)
        self.max_training_epochs = kwargs.get('max_training_epochs', 500)
        self.text_encoder, self.vae, self.unet, self.tokenizer = self.load_args_for_training()

    def load_args_for_training(self):
        text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="vae"
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="unet"
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        return text_encoder, vae, unet, tokenizer

    def __call__(self, text_encoder: CLIPTokenizer, vae: AutoencoderKL, unet: UNet2DConditionModel):
        logger = get_logger(__name__)
        set_seed(self.seed)
        HfFolder.save_token(self.token)
        api = HfApi()

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
        )

        if self.train_text_encoder and self.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training."
            )

        vae.requires_grad_(False)
        if not self.train_text_encoder:
            text_encoder.requires_grad_(False)

        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if self.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        optimizer_class = bnb.optim.AdamW8bit if self.use_8bit_adam else torch.optim.AdamW
        params_to_optimize = (
            itertools.chain(unet.parameters(),
                            text_encoder.parameters()) if self.train_text_encoder else unet.parameters()
        )
        optimizer = optimizer_class(params_to_optimize, lr=self.learning_rate)

        noise_scheduler = DDPMScheduler.from_config(self.pretrained_model_name_or_path, subfolder="scheduler")
        train_dataset = DreamBoothTextToImageDataset(self.dataset, self.dataset_config_name,
                                                     self.dataset_column, tokenizer=self.tokenizer)

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # Stack the resized pixel values
            pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()

            # Pad input_ids
            input_ids = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding="max_length",
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length
            ).input_ids

            return {"input_ids": input_ids, "pixel_values": pixel_values}

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
        )

        lr_scheduler = get_scheduler(
            self.lr_scheduler, optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.max_train_steps * self.gradient_accumulation_steps,
        )

        models_to_prepare = [unet, optimizer, train_dataloader, lr_scheduler]
        if self.train_text_encoder:
            models_to_prepare.insert(1, text_encoder)
        prepared_models = accelerator.prepare(*models_to_prepare)
        unet, optimizer, train_dataloader, lr_scheduler = prepared_models[:4]
        if self.train_text_encoder:
            text_encoder = prepared_models[1]

        weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else (
            torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
        )
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.decoder.to("cuda")
        if not self.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)

        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")

        progress_bar = tqdm(range(self.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if self.with_prior_preservation:
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                        prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")
                        loss = loss + self.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        if global_step % (self.save_steps * 2) == 0:
                            if accelerator.is_main_process:
                                pipeline = StableDiffusionPipeline.from_pretrained(
                                    self.pretrained_model_name_or_path,
                                    unet=accelerator.unwrap_model(unet),
                                    text_encoder=accelerator.unwrap_model(text_encoder),
                                )
                                save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                                pipeline.save_pretrained(save_path)

                    logs = {"loss": loss.detach().item()}
                    progress_bar.set_postfix(**logs)

                    if global_step >= self.max_train_steps:
                        break

            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.init_trackers(self.output_dir)
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
            )
            pipeline.save_pretrained(self.output_dir)
            del pipeline

            if self.push_to_hub:
                api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )
            torch.cuda.empty_cache()

        accelerator.end_training()


    @classmethod
    def image_grid(cls, imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def text_to_image(self, prompt, **kwargs):
        pipeline = StableDiffusionPipeline.from_pretrained(self.output_dir).to('cuda')
        images = pipeline(prompt, num_inference_steps=kwargs.get('num_inference_steps', 50),
                          num_images_per_prompt=kwargs.get('num_images_per_prompt', 4))
        torch.cuda.empty_cache()
        return StableDiffusionTextToImageFineTune.image_grid(images, 4, 1)

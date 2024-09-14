import itertools
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image, ImageDraw
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, get_scheduler, StableDiffusionInpaintPipeline
from huggingface_hub import upload_folder
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from ..utils import DreamBoothInpaintingDataset


class StableDiffusionInpaintingFineTune:

    def __init__(self, pretrained_model_name_or_path, resolution, center_crop,
                 train_text_encoder, dataset, dataset_config_name,
                 dataset_row_name, instance_prompt, learning_rate,
                 max_training_steps, save_steps, train_batch_size,
                 gradient_accumulation_steps, max_grad_norm,
                 mixed_precision, gradient_checkpointing,
                 use_8bit_adam, seed, with_prior_preservation,
                 prior_loss_weight, sample_batch_size, class_data_dir,
                 class_prompt, num_class_images, lr_scheduler,
                 lr_warmup_steps, output_dir, push_to_hub,
                 repo_id, token, max_training_epochs):
        """
        Ініціалізує об'єкт для fine-tuning моделі Stable Diffusion на завданні inpainting.

        Args:
            pretrained_model_name_or_path (str): Шлях або ім'я попередньо натренованої моделі.
            resolution (int): Роздільна здатність зображень, до якої будуть масштабуватися вхідні дані.
            center_crop (bool): Використовувати центральний crop під час підготовки даних.
            train_text_encoder (bool): Визначає, чи потрібно тренувати текстовий енкодер.
            dataset (Dataset): Об'єкт датасету, який використовується для тренування.
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
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.resolution = resolution
        self.center_crop = center_crop
        self.train_text_encoder = train_text_encoder
        self.dataset = dataset
        self.dataset_config_name = dataset_config_name
        self.dataset_row_name = dataset_row_name
        self.instance_prompt = instance_prompt
        self.learning_rate = learning_rate
        self.max_training_steps = max_training_steps
        self.save_steps = save_steps
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.use_8bit_adam = use_8bit_adam
        self.seed = seed
        self.with_prior_preservation = with_prior_preservation
        self.prior_loss_weight = prior_loss_weight
        self.sample_batch_size = sample_batch_size
        self.class_data_dir = class_data_dir
        self.class_prompt = class_prompt
        self.num_class_images = num_class_images
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.output_dir = output_dir
        self.push_to_hub = push_to_hub
        self.repo_id = repo_id
        self.token = token
        self.max_training_epochs = max_training_epochs
        self.text_encoder, self.vae, self.unet, self.tokenizer = self.load_args_for_training()

    def prepare_mask_and_masked_image(self, image, mask):
        """
        Перетворює зображення та маску у тензори, а також готує зображення, на яке накладено маску.
        Створює датасет з оригінальним зображенням, маскою, бінаризованою маскою та замаскованим зображенням.

        Args:
            image (PIL.Image): Вхідне зображення у форматі PIL.
            mask (PIL.Image): Маска зображення у форматі PIL.

        Returns:
            dict: Словник з оригінальним зображенням, маскою, бінаризованою маскою та замаскованим зображенням у форматі PyTorch тензорів.
        """

        # Перетворюємо зображення у формат RGB
        image = np.array(image.convert("RGB"))

        # Додаємо вимірювання для батчу та змінюємо порядок вимірів
        image = image[None].transpose(0, 3, 1, 2)

        # Перетворюємо у тензор PyTorch та нормалізуємо
        image_tensor = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # Перетворюємо маску в градації сірого та конвертуємо у масив NumPy
        mask = np.array(mask.convert("L"))

        # Нормалізуємо маску
        mask = mask.astype(np.float32) / 255.0

        # Додаємо вимірювання для батчу та каналу
        mask = mask[None, None]

        # Бінаризуємо маску
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Перетворюємо маску у тензор PyTorch
        mask_tensor = torch.from_numpy(mask)

        # Створюємо замасковане зображення
        masked_image_tensor = image_tensor * (mask_tensor < 0.5)

        # Повертаємо датасет у вигляді словника
        dataset = {
            'image': image_tensor,
            'mask': mask_tensor,
            'binary_mask': mask_tensor,
            'masked_image': masked_image_tensor
        }

        return dataset

    def random_mask(self, im_shape, ratio=1, mask_full_image=False):
        # Створюємо нову чорно-білу маску розміром im_shape, заповнену чорним кольором (0).
        mask = Image.new("L", im_shape, 0)

        # Створюємо об'єкт для малювання на масці.
        draw = ImageDraw.Draw(mask)

        # Визначаємо випадковий розмір фігури (прямокутник або еліпс),
        # де розміри залежать від заданого коефіцієнта ratio.
        size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))

        # Якщо встановлено mask_full_image=True, фігура має бути такого ж розміру, як і все зображення.
        if mask_full_image:
            size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))

        # Визначаємо межі, в яких можна розмістити центр фігури, щоб вона повністю вміщувалася в маску.
        limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)

        # Визначаємо випадкове положення центру фігури в межах маски.
        center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))

        # Випадково вибираємо тип фігури: 0 — прямокутник, 1 — еліпс.
        draw_type = random.randint(0, 1)

        # Якщо обрано прямокутник або якщо маска має повністю покривати зображення:
        if draw_type == 0 or mask_full_image:
            # Малюємо прямокутник білого кольору (255) в масці.
            draw.rectangle(
                (
                center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
                fill=255,
            )
        else:
            # Інакше малюємо еліпс білого кольору (255) в масці.
            draw.ellipse(
                (
                center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
                fill=255,
            )

        # Повертаємо згенеровану маску.
        return mask

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

    def collate_fn(self, examples):
        """
        Обробляє та підготовляє батч даних для моделі, включаючи маски та замасковані зображення.

        Args:
            examples (list): Список прикладів, де кожен приклад містить вхідні дані, такі як зображення, маски та ідентифікатори підказок.

        Returns:
            dict: Повертає батч даних у вигляді словника, що містить ідентифікатори підказок, значення пікселів зображень, маски та замасковані зображення.
        """

        # Отримуємо списки input_ids та зображень з кожного прикладу.
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Додаємо приклади класів для збереження пріоритету (якщо це зазначено у конфігурації),
        # щоб уникнути двох проходів вперед.
        if self.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            pior_pil = [example["class_PIL_images"] for example in examples]

        # Ініціалізуємо списки для збереження масок та замаскованих зображень.
        masks = []
        masked_images = []

        # Обробляємо кожен приклад.
        for example in examples:
            pil_image = example["PIL_images"]

            # Генеруємо випадкову маску або використовуємо надану маску, якщо вона є.
            if example.get('PIL_Mask'):
                mask = example["PIL_Mask"]
            else:
                mask = self.random_mask(pil_image.size, 1, False)

            # Готуємо маску та замасковане зображення за допомогою функції prepare_mask_and_masked_image.
            mask, masked_image = self.prepare_mask_and_masked_image(pil_image, mask)

            # Додаємо маску та замасковане зображення до відповідних списків.
            masks.append(mask)
            masked_images.append(masked_image)

        # Якщо збереження пріоритету ввімкнено, додатково обробляємо зображення класів.
        if self.with_prior_preservation:
            for pil_image in pior_pil:
                # Генеруємо випадкову маску для зображення.
                mask = self.random_mask(pil_image.size, 1, False)

                # Готуємо маску та замасковане зображення.
                mask, masked_image = self.prepare_mask_and_masked_image(pil_image, mask)

                # Додаємо їх до списків.
                masks.append(mask)
                masked_images.append(masked_image)

        # Об'єднуємо всі зображення у тензор PyTorch.
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # Об'єднуємо input_ids в тензор, додаючи паддінг там, де необхідно.
        input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        # Об'єднуємо маски та замасковані зображення у відповідні тензори PyTorch.
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)

        # Створюємо та повертаємо батч у вигляді словника.
        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "masks": masks,
            "masked_images": masked_images
        }

        return batch

    def __call__(self, *args):
        logger = get_logger(__name__)

        set_seed(self.seed)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
        )

        # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
        # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
        # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
        if self.train_text_encoder and self.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        self.vae.requires_grad_(False)
        if not self.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = (
            itertools.chain(self.unet.parameters(),
                            self.text_encoder.parameters()) if self.train_text_encoder else self.unet.parameters()
        )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.learning_rate,
        )

        noise_scheduler = DDPMScheduler.from_config(self.pretrained_model_name_or_path, subfolder="scheduler")

        train_dataset = DreamBoothInpaintingDataset(
            dataset_name=self.dataset,
            config=self.dataset_config_name,
            train_name=self.dataset_row_name,
            instance_prompt=self.instance_prompt,
            class_data_root=self.class_data_dir if self.with_prior_preservation else None,
            class_prompt=self.class_prompt,
            tokenizer=self.tokenizer,
            size=self.resolution,
            center_crop=self.center_crop,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True, collate_fn=self.collate_fn
        )

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.max_training_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.max_training_steps * self.gradient_accumulation_steps,
        )

        if self.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.unet, self.text_encoder, self.optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.unet, self.optimizer, train_dataloader, lr_scheduler
            )

        accelerator.register_for_checkpointing(lr_scheduler)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.vae.decoder.to("cuda")
        if not self.train_text_encoder:
            self.text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth", config=vars(args))

        # Train!
        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.max_training_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, self.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")


        for epoch in range(self.max_training_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                #     if step % args.gradient_accumulation_steps == 0:
                #         progress_bar.update(1)
                #     continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space

                    latents = self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Convert masked images to latent space
                    masked_latents = self.vae.encode(
                        batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    masked_latents = masked_latents * self.vae.config.scaling_factor

                    masks = batch["masks"]
                    # resize the mask to latents shape as we concatenate the mask to the latents
                    mask = torch.stack(
                        [
                            torch.nn.functional.interpolate(mask, size=(self.resolution // 8, self.resolution // 8))
                            for mask in masks
                        ]
                    )
                    mask = mask.reshape(-1, 1, self.resolution // 8, self.resolution // 8)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # concatenate the noised latents with the mask and the masked latents
                    latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if self.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + self.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if self.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
            )
            pipeline.save_pretrained(self.output_dir)

            if self.push_to_hub:
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        accelerator.end_training()
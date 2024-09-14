# FineTuning

This project provides a toolkit for fine-tuning the Stable Diffusion model for inpainting tasks (image restoration based on a mask) using PyTorch and Hugging Face Diffusers libraries.

## Requirements

Before starting, make sure to install the following libraries:

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

* torch
* diffusers
* transformers
* accelerate
* huggingface_hub
* PIL
* numpy
* tqdm 

## Description

The StableDiffusionInpaintingFineTune class is responsible for fine-tuning the Stable Diffusion model for inpainting tasks. It supports training both the text encoder and the UNet model and allows various settings to control the training process.

## Constructor

```
__init__(...)
```
* `pretrained_model_name_or_path`: The path or name of the pre-trained model.
* `resolution`: The resolution of the images.
* `center_crop`: Whether to apply center cropping during data preparation.
* `train_text_encoder`: Whether to train the text encoder.
* `dataset`: The dataset object.
* `learning_rate`: The initial learning rate.
* `max_training_steps`: The maximum number of training steps.
* `save_steps`: The number of steps between saving checkpoints.
* `train_batch_size`: The batch size.
* `gradient_accumulation_steps`: The number of steps to accumulate gradients.
* `mixed_precision`: Use of mixed precision ("fp16", "bf16", or None).
* `gradient_checkpointing`: Use of gradient checkpointing.
* `use_8bit_adam`: Use of the 8-bit Adam optimizer.
* `seed`: The random seed for reproducibility.
* `output_dir`: The directory for saving results.
* `push_to_hub`: Whether to upload the results to the Hugging Face Hub.
* `repo_id`: The repository ID on Hugging Face Hub.

## Methods

`prepare_mask_and_masked_image`(`image`, `mask`): Prepares the mask and masked image.
`random_mask`(`im_shape`, `ratio`=1, `mask_full_image`=False): Generates a random mask.
`load_args_for_training`(): Loads the necessary components of the model for training.
`collate_fn`(`examples`): Forms a batch of data for the model.
`__call__`(self, *args, **kwargs): The main method for running the training process.

## Usage

To start training, create an instance of the StableDiffusionInpaintingFineTune class and call its __call__ method with the required arguments.



```python
from src.dreamfinetune import StableDiffusionInpaintingFineTune

model = StableDiffusionInpaintingFineTune(
    pretrained_model_name_or_path="path_to_model",
    resolution=512,
    center_crop=True,
)
model()
```


License

This project is distributed under the MIT License.


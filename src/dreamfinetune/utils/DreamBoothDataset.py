import shutil
from pathlib import Path

import torch
from PIL import Image as PILImage
from datasets import load_dataset, Image, DatasetInfo, Dataset as Datasets, Split
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from transformers import CLIPTokenizer


class DreamBoothInpaintingDataset(Dataset):
    def __init__(
            self,
            dataset_name: str,
            config: str,
            instance_prompt: str,
            tokenizer: CLIPTokenizer = None,
            class_data_root=None,
            class_prompt: str = None,
            size: int = 1024,
            center_crop: bool = False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_dataset = load_dataset(dataset_name, config)
        # if not self.instance_data_root.exists():
        #     raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = self.instance_dataset['train']['images']
        self.instance_masks_path = self.instance_dataset['train']['mask']
        self.binary_mask = self.instance_dataset['train']['binary_mask']
        self.binary_masked_image = self.instance_dataset['train']['masked_image']

        self.num_instance_images = len(self.instance_images_path)
        self.num_instance_masks = len(self.instance_masks_path)

        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_path[index]
        instance_mask = self.instance_masks_path[index]
        instance_binary_mask = self.binary_mask[index]
        instance_binary_masked_image = self.binary_masked_image[index]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        mask = instance_mask.convert("L")
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        example["PIL_images"] = instance_image
        example["PIL_masks"] = mask

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_mask"] = self.image_transforms(mask)

        example["binary_mask"] = instance_binary_mask
        example["masked_image"] = instance_binary_masked_image

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = PILImage.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms_resize_and_crop(class_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class DreamBoothTextToImageDataset(Dataset):
    def __init__(self, dataset_name: str, config: str, row: str, tokenizer: CLIPTokenizer):
        self.row = row
        self.instance_dataset = DreamBoothTextToImageDataset.load(dataset_name, config)
        self._length = len(self.instance_dataset[row])
        self.tokenizer = tokenizer


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        instance_prompt_ids = self.tokenizer(
            self.instance_dataset['instance_prompt'][index],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        instance_images = image_transform(self.instance_dataset['PIL_Image'][index])
        return {
            "PIL_images": self.instance_dataset['PIL_Image'][index],
            "instance_images": instance_images,
            "instance_prompt_ids": instance_prompt_ids
        }

    @classmethod
    def create_dataset(cls, citation: str, PIL_images: list[str], instance_prompt: str, size,
                       tokenizer: CLIPTokenizer, center_crop=False):
        dataset_info = DatasetInfo(
            description="This dataset contains Images, Tokenization Prompt and Tensor Image for training text-to-image.",
            license="MIT",
            citation=citation
        )

        resize_crop_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        ])

        images = []

        for id, image_path in enumerate(PIL_images):
            image = PILImage.open(image_path)
            image = resize_crop_transform(image)
            save_path = Path(image_path).parent / 'data' / f'{id}.png'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            image.save(save_path)
            images.append(str(save_path))

        instance_prompt_ids = tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length
        ).input_ids

        dataset = Dataset.from_dict({
            'PIL_Image': images,
            'instance_prompt_ids': [instance_prompt_ids] * len(images)
        }, info=dataset_info).cast_column('PIL_Image', Image())

        return dataset

    @classmethod
    def load(cls, dataset_path: str, config_name: str):
        return load_dataset(dataset_path, config_name, split='train')

    @classmethod
    def push_to_hub(cls, dataset: Datasets, repo_id: str, config_name: str, private: bool, token: str):
        dataset.push_to_hub(repo_id=repo_id, config_name=config_name, private=private, token=token)

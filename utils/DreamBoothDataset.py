from pathlib import Path

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DreamBoothInpaintDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            config,
            instance_prompt,
            tokenizer=None,
            class_data_root=None,
            class_prompt=None,
            size=1024,
            center_crop=False,
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
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
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
    def __init__(
            self,
            dataset_name,
            config,
            row,
            instance_prompt,
            tokenizer=None,
            class_data_root=None,
            class_prompt=None,
            size=1024,
            center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_dataset = load_dataset(dataset_name, config)
        # if not self.instance_data_root.exists():
        #     raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = self.instance_dataset['train'][row]

        self.num_instance_images = len(self.instance_images_path)


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

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        instance_image = self.image_transforms_resize_and_crop(instance_image)

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
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

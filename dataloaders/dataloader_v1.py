import os
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from wcmatch.pathlib import Path
import numpy as np

from utils.utils import fft2d, zero_padding_twice

__DATASETS__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASETS__.get(name, None):
            raise NameError(f"Name {name} is already registered.")
        __DATASETS__[name] = cls
        return cls

    return wrapper


def get_dataset(name: str):
    if __DATASETS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASETS__[name]

    
def get_loader(dataset_name: str, stage: str, root: str, batch_size: int, **kwargs):
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Grayscale(),
        ]
    )
    dataset = get_dataset(dataset_name)(root, stage, transform=transform, **kwargs)
    data_loader = DataLoader(dataset, batch_size)
    return data_loader


@register_dataset(name="png_dataset")
class PNGDataset(VisionDataset):
    def __init__(self, root: str, stage: str, transform):
        super().__init__(root=root)
        self.transform = transform
        self.image_paths = sorted(
            (Path(root) / Path(stage)).rglob(["*.png", "*.jpg", "*.jpeg", "*.bmp"])
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        image = Image.open(fp=img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image
    
@register_dataset(name="adversarial_dataset")
class AdversarialDataset():
    def __init__(
        self, root: str, stage: str, transform: Optional[Callable] = None
    ):
        self.gt_images = PNGDataset(root, f"{stage}/gt", transform)
        self.init_images = PNGDataset(root, f"{stage}/robust_hio", transform)
        self.output_images = PNGDataset(root, f"{stage}/output", transform)
        
    def __len__(self):
        return len(self.gt_images)
    
    def __getitem__(self, index):
        return self.gt_images[index], self.init_images[index], self.output_images[index]


@register_dataset(name="adversarial_dataset_large")
class AdversarialDatasetLarge():
    def __init__(
        self, root: str, stage: str, alpha: float, transform: Optional[Callable] = None
    ):
        hio_folder = Path(root) / Path(f"{stage}_outputs_large_alpha_{alpha}")
        self.hio_image_paths = sorted(
            hio_folder.rglob(["*.npy"])
        )
        self.gt_image_paths = [
            Path(root) / stage / Path(image_path).relative_to(hio_folder).with_suffix("")
            for image_path in self.hio_image_paths
        ]
        self.transform = transform
        
    def __len__(self):
        return len(self.hio_image_paths)
    
    def __getitem__(self, index):
        img_path = self.gt_image_paths[index]
        image = Image.open(fp=img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(np.load(self.hio_image_paths[index]))



@register_dataset(name="amplitude_no_pad_dataset")
class AmplitudeNoPadDataset(PNGDataset):
    def __getitem__(self, index):
        image = super().__getitem__(index)
        fft_image = fft2d(image)
        amplitude = fft_image.abs()
        return image, amplitude, image


@register_dataset(name="amplitude_dataset")
class AmplitudeDataset(PNGDataset):
    def __getitem__(self, index):
        image = super().__getitem__(index)

        support = torch.ones_like(image)
        image = zero_padding_twice(image)
        support = zero_padding_twice(support)

        fft_image = fft2d(image)
        amplitude = fft_image.abs()

        return image, amplitude, support


@register_dataset(name="noise_amplitude_dataset")
class NoiseAmplitudeDataset(AmplitudeDataset):
    def __init__(
        self,
        root: str,
        stage: str,
        alpha: float,
        transform: Optional[Callable] = None,
        return_noiseless: bool = False,
    ):
        super().__init__(root, stage, transform)
        self.alpha = alpha
        self.return_noiseless = return_noiseless

    def __getitem__(self, index):
        image, amplitude, support = super().__getitem__(index)

        intensity_noise = self.alpha * amplitude * torch.randn(amplitude.shape)
        y2 = torch.square(amplitude) + intensity_noise
        y2 = torch.max(y2, torch.zeros_like(y2))
        amplitude_noisy = torch.sqrt(y2)

        if self.return_noiseless:
            return image, amplitude_noisy, support, amplitude
        else:
            return image, amplitude_noisy, support

@register_dataset(name="noise_amplitude_with_robust_hio_dataset")
class NoiseAmplitudeWithRobustHIODataset(AmplitudeDataset):
    def __init__(
        self, root: str, stage: str, alpha: float, transform: Optional[Callable] = None
    ):
        super().__init__(root, stage, transform)
        self.alpha = alpha
        self.robust_hio_dataset = PNGDataset(
            root, f"{stage}_robust_hio_alpha_{alpha}", transform
        )

    def __len__(self):
        return len(self.robust_hio_dataset)

    def __getitem__(self, index):
        image, amplitude, support = super().__getitem__(index)

        intensity_noise = self.alpha * amplitude * torch.randn(amplitude.shape)
        y2 = torch.square(amplitude) + intensity_noise
        y2 = torch.max(y2, torch.zeros_like(y2))
        amplitude = torch.sqrt(y2)

        robust_hio_output = self.robust_hio_dataset[index]

        return image, amplitude, support, robust_hio_output

@register_dataset(name="noise_amplitude_with_multiple_hio_dataset")
class NoiseAmplitudeWithMultipleHIODataset(VisionDataset):
    def __init__(
        self,
        root: str,
        stage: str,
        alpha: list,
        transform: Optional[Callable] = None,
    ):
        self.alpha = alpha
        self.transform = transform
        hio_folder = Path(root) / Path(f"{stage}_multioutput_robust_hio_alpha_{alpha}")
        self.hio_image_paths = sorted(
            hio_folder.rglob(["*.npy"])
        )
        self.gt_image_paths = [
            Path(root) / stage / Path(image_path).relative_to(hio_folder).with_suffix("")
            for image_path in self.hio_image_paths
        ]

    def __len__(self):
        return len(self.hio_image_paths)

    def __getitem__(self, index: int):
        img_path = self.gt_image_paths[index]
        image = Image.open(fp=img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        
        hio_output = torch.tensor(np.load(self.hio_image_paths[index]))
        
        support = torch.ones_like(image)
        image = zero_padding_twice(image)
        support = zero_padding_twice(support)

        fft_image = fft2d(image)
        amplitude = fft_image.abs()

        intensity_noise = self.alpha * amplitude * torch.randn(amplitude.shape)
        y2 = torch.square(amplitude) + intensity_noise
        y2 = torch.max(y2, torch.zeros_like(y2))
        amplitude_noisy = torch.sqrt(y2)

        return image, amplitude_noisy, amplitude, support, hio_output
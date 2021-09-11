import json

import torch
import torchvision

from config.data_mode import Mode


class ClassificationConfig:
    num_epochs = 21
    lr = 1e-4
    num_classes = 4
    validation_frequency = 2
    batch_size = 12
    num_workers = 6
    device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(
        torch.cuda.current_device()).total_memory > 4e9 else "cpu"
    path = {
        Mode.train: "./data/OCT/train",
        Mode.eval: "./data/OCT/val"
    }

    transforms = {
        Mode.train: torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]),
        Mode.eval: torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]),
    }

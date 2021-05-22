import json

import torch


class ClassificationConfig:
    num_heads = 8
    in_channels = 3
    embed_size = 768
    num_classes = 2
    num_layers = 6
    image_shape = 224
    patch_size = 16
    num_epochs = 1
    lr = 1e-3

    validation_frequency = 2
    batch_size = 4
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

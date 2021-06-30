import torch.utils.data as torchdata

from config.dino_config import DINOConfig


class DinoDataset(torchdata.Dataset):

    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def get_data_loaders(config: DINOConfig):
    pass

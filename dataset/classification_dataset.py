import torch.utils.data as data
from torchvision.datasets import ImageFolder

from config.classification_config import ClassificationConfig
from config.data_mode import Mode


def get_data_loaders(config: ClassificationConfig):

    dataset_train = ImageFolder(root=config.path[Mode.train],
                                transform=config.transforms[Mode.train])
    dataset_eval = ImageFolder(root=config.path[Mode.eval],
                               transform=config.transforms[Mode.eval])

    return {
        Mode.train: data.DataLoader(dataset_train, batch_size=config.batch_size,
                                    shuffle=True, num_workers=config.num_workers),
        Mode.eval: data.DataLoader(dataset_eval, batch_size=config.batch_size,
                                   shuffle=True, num_workers=config.num_workers)
    }

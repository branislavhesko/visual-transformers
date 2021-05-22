import torch.utils.data as data
from torchvision.datasets import ImageFolder

from config.data_mode import Mode


def get_data_loaders(config):

    dataset_train = ImageFolder(root=config.path[Mode.train])
    dataset_eval = ImageFolder(root=config.path[Mode.eval])

    return {
        Mode.train: data.DataLoader(dataset_train, batch_size=config.batch_size,
                                    shuffle=True, num_workers=config.num_workers),
        Mode.eval: data.DataLoader(dataset_eval, batch_size=config.batch_size,
                                   shuffle=True, num_workers=config.num_workers)
    }

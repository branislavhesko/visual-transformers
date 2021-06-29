import torch
from matplotlib import pyplot as plt

from config.vit_config import VITConfig, Mode
from dataset.classification_dataset import get_data_loaders
from modeling.vit import VIT
from utils.visualization import get_attentions, visualize

config = VITConfig()
model = VIT(
            num_layers=config.num_layers,
            in_channels=config.in_channels,
            embed_size=config.embed_size,
            num_classes=config.num_classes,
            num_heads=config.num_heads,
            image_shape=config.image_shape,
            patch_size=config.patch_size,
            store_attention=True
        ).to(config.device)
model.load_state_dict(torch.load("ckpt_mine.pth"))
model.eval()
loaders = get_data_loaders(config)


for idx in range(1000):
    image = loaders[Mode.train].dataset[idx][0].to(config.device).unsqueeze(0)
    attentions = get_attentions(model, image)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image.cpu().squeeze().permute(1, 2, 0).numpy() * std + mean
    mask = visualize(attentions, image)

    plt.subplot(1, 2, 1)
    plt.imshow(mask * image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()

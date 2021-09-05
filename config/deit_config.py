from config.classification_config import ClassificationConfig
from modeling.vit import VIT


class DeITConfig(ClassificationConfig):

    checkpoint_path = "./ckpt/checkpoint.pt"

    def __init__(self):
        self.aux_model = VIT(
            num_layers=self.num_layers,
            in_channels=self.in_channels,
            embed_size=self.embed_size,
            num_classes=self.num_classes,
            num_heads=self.num_heads,
            image_shape=self.image_shape,
            patch_size=self.patch_size,
            store_attention=False
        )

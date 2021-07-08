import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
import tqdm

from config.deit_config import DeITConfig
from config.vit_config import VITConfig
from config.data_mode import Mode
from dataset.classification_dataset import get_data_loaders
from modeling.deit import DeIT, get_auxiliary_model


class DEITTrainer:

    def __init__(self, config: DeITConfig):
        self.config = config
        self._model = DeIT(
            num_layers=self.config.num_layers,
            in_channels=self.config.in_channels,
            embed_size=self.config.embed_size,
            num_classes=self.config.num_classes,
            num_heads=self.config.num_heads,
            image_shape=self.config.image_shape,
            patch_size=self.config.patch_size,
            store_attention=False
        ).to(self.config.device)
        self._aux_model = get_auxiliary_model(self.config.aux_model, self.config.checkpoint_path).to(self.config.device)
        self._loss = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=self.config.lr, weight_decay=1e-4)
        self._scheduler = optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=0.95)
        self._data_loader = get_data_loaders(self.config)
        self._writer = tb.SummaryWriter()

    def train(self):
        for epoch in range(self.config.num_epochs):
            self._train_step(epoch)

    def _train_step(self, epoch):
        loader = tqdm.tqdm(self._data_loader[Mode.train])

        for sample_idx, data in enumerate(loader):
            self._optimizer.zero_grad()
            images, labels = [d.to(self.config.device) for d in data]

            aux_output = self._aux_model(images)
            classification_output, distillation_output = self._model(images)
            loss = 0.5 * (self._loss(classification_output, labels) + self._loss(distillation_output, aux_output))
            loss.backward()
            self._optimizer.step()
            self._writer.add_scalar("TrainingLoss", loss.item(),
                                    epoch * len(self._data_loader[Mode.train]) + sample_idx)

    def validate(self, epoch):
        pass

    def _visualize_attentions(self):
        pass


if __name__ == "__main__":
    DEITTrainer(DeITConfig()).train()

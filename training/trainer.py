import torch
import tqdm

from config.classification_config import ClassificationConfig
from config.data_mode import Mode
from dataset.classification_dataset import get_data_loaders
from modeling.vit import VIT


class Trainer:

    def __init__(self, config: ClassificationConfig):
        self.config: ClassificationConfig = config
        self._model = VIT(
            num_layers=self.config.num_layers,
            in_channels=self.config.in_channels,
            embed_size=self.config.embed_size,
            num_classes=self.config.num_classes,
            num_heads=self.config.num_heads,
            image_shape=self.config.image_shape,
            patch_size=self.config.patch_size
        )
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self.config.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=0.9)
        self._loss = torch.nn.CrossEntropyLoss()
        self._loader = get_data_loaders(self.config)

    def train(self):

        for epoch in range(self.config.num_epochs):
            self._train_single_epoch(epoch)

            if epoch % self.config.validation_frequency == 0:
                self.validate(epoch)

    def _train_single_epoch(self, epoch):
        progress = tqdm.tqdm(self._loader[Mode.train])

        for index, data in enumerate(progress):
            self._optimizer.zero_grad()
            data = [d.to(self.config.device) for d in data]
            images, labels = data
            output = self._model(images)
            loss = self._loss(output, labels)
            loss.backward()
            self._optimizer.step()
            prediction = output.argmax(dim=-1)
            progress.set_description("Epoch: {}, Loss: {:.2f}, Prediction: {}, Labels: {}".format(
                epoch, loss.item(), prediction, labels
            ))

    def validate(self, epoch):
        pass

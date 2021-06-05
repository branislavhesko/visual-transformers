import torch
import torch.utils.tensorboard as tb
import tqdm

from config.classification_config import ClassificationConfig
from config.data_mode import Mode
from dataset.classification_dataset import get_data_loaders
from vit import ViT
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
            patch_size=self.config.patch_size,
            store_attention=False
        ).to(self.config.device)
        # self._model = ViT(
        #     image_size=256,
        #     patch_size=32,
        #     num_classes=4,
        #     dim=1024,
        #     depth=6,
        #     heads=16,
        #     mlp_dim=2048,
        #     dropout=0.0,
        #     emb_dropout=0.0
        # ).to(self.config.device)
        # self._model.load_state_dict(torch.load("./ckpt.pth"))
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self.config.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=0.95)
        self._loss = torch.nn.CrossEntropyLoss()
        self._loader = get_data_loaders(self.config)
        self._writer = tb.SummaryWriter()

    def train(self):

        for epoch in range(self.config.num_epochs):
            self._train_single_epoch(epoch)

            if epoch % self.config.validation_frequency == 0:
                self.validate(epoch)
            torch.save(self._model.state_dict(), "ckpt_mine.pth")
            self._scheduler.step()

    def _train_single_epoch(self, epoch):
        progress = tqdm.tqdm(self._loader[Mode.train])
        ok = 0
        total = 0
        for index, data in enumerate(progress):
            self._optimizer.zero_grad()
            data = [d.to(self.config.device) for d in data]
            images, labels = data
            output = self._model(images)
            loss = self._loss(output, labels)
            loss.backward()
            self._optimizer.step()
            prediction = output.argmax(dim=-1)
            ok += (prediction == labels).sum()
            total += len(prediction)
            progress.set_description("Epoch: {}, Loss: {:.2f}, Prediction: {}, Labels: {}, OK: {}".format(
                epoch, loss.item(), prediction, labels, (prediction == labels).sum()
            ))
            self._writer.add_scalar("Loss", loss.item(), index + epoch * len(self._loader[Mode.train]))
        print("Accuracy: {}".format(ok / total))

    def validate(self, epoch):
        pass


if __name__ == "__main__":
    trainer = Trainer(ClassificationConfig())
    trainer.train()
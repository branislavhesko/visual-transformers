import torch
import torch.utils.tensorboard as tb
import tqdm

from config.vit_config import VITConfig
from config.data_mode import Mode
from dataset.classification_dataset import get_data_loaders
from model_config import swin_small_window_7_embed_96

class Trainer:

    def __init__(self, config: VITConfig):
        self.config: VITConfig = config
        self._model = swin_small_window_7_embed_96(self.config.num_classes).cuda()
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
            torch.save(self._model.state_dict(), "ckpt_{}.pth".format(self._model.__class__.__name__))
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
    trainer = Trainer(VITConfig())
    trainer.train()
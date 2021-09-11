import torch
import torch.utils.tensorboard as tb
import tqdm

from config.classification_config import ClassificationConfig
from config.data_mode import Mode
from dataset.classification_dataset import get_data_loaders
from model_config import swin_small_window_7_embed_96
from swin import SwinTransformer



class Trainer:

    def __init__(self, config: ClassificationConfig):
        self.config: ClassificationConfig = config
        self._model = swin_small_window_7_embed_96(num_classes=config.num_classes).cuda()
        # self._model.load_state_dict(torch.load("./ckpt.pth"))
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.config.lr, weight_decay=5e-2)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=0.95)
        self._loss = torch.nn.CrossEntropyLoss()
        self._loader = get_data_loaders(self.config)
        self._writer = tb.SummaryWriter()

    def train(self):

        for epoch in range(self.config.num_epochs):
            self._train_single_epoch(epoch)

            # if epoch % self.config.validation_frequency == 0:
            #     self.validate(epoch)
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
            progress.set_description("Epoch: {}, Loss: {:.2f}, Prediction: {}, Labels: {}, OK: {}, Accuracy: {}".format(
                epoch, loss.item(), prediction, labels, (prediction == labels).sum(), ok / total
            ))
            self._writer.add_scalar("Loss", loss.item(), index + epoch * len(self._loader[Mode.train]))
        print("Accuracy: {}".format(ok / total))

    def validate(self, epoch):
        self._model.eval()
        progress = tqdm.tqdm(self._loader[Mode.eval])

        for batch_idx, data in enumerate(progress):
            image, labels = [d.to(self.config.device) for d in data]
            output = self._model(image)
            loss = self._loss(output, labels)
            prediction = output.argmax(dim=1)
            # TODO: finish

if __name__ == "__main__":
    trainer = Trainer(ClassificationConfig())
    trainer.train()
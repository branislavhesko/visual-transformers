import copy

import torch
import torch.nn as nn
import torch.optim as optim

from config.dino_config import DINOConfig
from modeling.dino_loss import DINOLoss


class DINOTrainer:

    def __init__(self, config: DINOConfig):
        self.config = config
        self.student = torch.nn.Module()
        self.teacher = copy.deepcopy(self.student)
        self.optimizer = optim.SGD(self.student.parameters(), lr=self.config.lr,
                                   weight_decay=1e-4, momentum=0.95, nesterov=True)
        self.loss = DINOLoss(weights_momentum=self.config.weights_momentum, center_momentum=self.config.center_momentum, temperature=self.config.temperature)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.gamma)

    def train(self):
        for epoch in range(self.config.num_epochs):
            self._train_one_epoch(epoch)

    def _single_step(self, images):
        si1, si2 = [self.student(image) for image in images]
        with torch.no_grad():
            ti1, ti2 = [self.teacher(image) for image in images]
        loss = (self.loss(si1, ti2) + self.loss(si2, ti1)) / 2.
        loss.backward()
        self.optimizer.step()

    def _train_one_epoch(self, epoch):
        pass

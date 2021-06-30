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
        student_output = [self.student(image) for image in images]
        with torch.no_grad():
            teacher_output = [self.teacher(image) for image in images]
        loss = self.loss(student_output=student_output, teacher_output=teacher_output)
        apply_ema(self.teacher, self.student, decay=self.config.decay)
        loss.backward()
        self.optimizer.step()

    def _train_one_epoch(self, epoch):
        pass

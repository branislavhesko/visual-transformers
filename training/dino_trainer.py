import copy
from typing import Dict

import torch
import torch.utils.data.dataloader as dl
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.data_mode import Mode
from config.dino_config import DINOConfig
from dataset.dino_dataset import get_data_loaders
from modeling.dino_loss import DINOLoss
from modeling.ops import apply_ema


class DINOTrainer:

    def __init__(self, config: DINOConfig):
        self.config = config
        self.student = torch.nn.Module()
        self.teacher = copy.deepcopy(self.student)
        self.optimizer = optim.SGD(self.student.parameters(), lr=self.config.lr,
                                   weight_decay=1e-4, momentum=0.95, nesterov=True)
        self.loss = DINOLoss(weights_momentum=self.config.weights_momentum,
                             center_momentum=self.config.center_momentum, temperature=self.config.temperature)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.gamma)
        self.writer = SummaryWriter()
        self.data_loader: Dict[Mode, dl.DataLoader] = get_data_loaders(config=self.config)

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_one_epoch(epoch)

            if epoch % self.config.validation_frequency == 0:
                self.validate(epoch)
                self.save(self.config.model_name)

    def save(self, model_name):
        pass

    def validate(self, epoch):
        pass

    def _single_step(self, images):
        student_output = [self.student(image) for image in images]
        with torch.no_grad():
            teacher_output = [self.teacher(image) for image in images]
        loss = self.loss(student_output=student_output, teacher_output=teacher_output)
        apply_ema(self.teacher, self.student, decay=self.config.decay)
        loss.backward()
        self.optimizer.step()
        return teacher_output, loss

    def train_one_epoch(self, epoch):
        progress_bar = tqdm(self.data_loader[Mode.train])

        for batch_idx, data in enumerate(progress_bar):
            self.optimizer.zero_grad()
            data = [d.to(self.config.device) for d in data]
            teacher_output, loss = self._single_step(data)
            progress_bar.set_description(f"EPOCH: {epoch}, loss: {loss.item():.2f}")
            if batch_idx % self.config.visualization_frequency[Mode.train] == 0:
                self._visualize_attention(attentions=teacher_output)

    def _visualize_attention(self, attentions):
        pass

import torch
from torch import nn
import torchvision
import torchmetrics


class PneumoniaResNet152NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet152()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=1, bias=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
    def forward(self, data):
        return self.model(data)

    def training_step(self, batch):
        x_ray, label = batch
        x_ray = x_ray.float()
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)
        acc = self.train_acc(torch.sigmoid(pred), label.int())

        print(f"Train Loss: {loss.item():.4f}")
        print(f"Step Train ACC: {acc:.4f}")

        return loss

    def on_train_epoch_end(self):
        self.log("Training ACC", self.train_acc.compute())

    def validation_step(self, batch):
        x_ray, label = batch
        x_ray = x_ray.float()
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)
        acc = self.val_acc(torch.sigmoid(pred), label)

        print("VAL Loss", loss)
        print("Step VAL ACC", acc)

        return loss

    def on_validation_epoch_end(self):
        print("VAL ACC", self.val_acc.compute())

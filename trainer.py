import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = torch.sigmoid(outputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        return running_loss / len(dataloader.dataset)

    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        return running_loss / len(dataloader.dataset)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")



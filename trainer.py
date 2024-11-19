import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, device, checkpoints_path="checkpoints"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoints_path = checkpoints_path
        self.top_losses = []
        self.start_epoch = 0

    def save_top_checkpoints(self, epoch, loss):
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        if len(self.top_losses) < 5 or loss < max(self.top_losses, key=lambda x: x[0])[0]:
            if len(self.top_losses) == 5:
                self.top_losses = sorted(self.top_losses, key=lambda x: x[0])
                worst_loss, worst_epoch = self.top_losses.pop()
                worst_path = f"{self.checkpoints_path}/best_epoch_{worst_epoch}_loss_{worst_loss:.4f}.pt"
                if os.path.exists(worst_path):
                    os.remove(worst_path)

            self.top_losses.append((loss, epoch))
            checkpoint_path = f"{self.checkpoints_path}/best_epoch_{epoch}_loss_{loss:.4f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            }, checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch} with loss {loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: {checkpoint_path} (Epoch: {self.start_epoch}, Loss: {checkpoint['loss']:.4f})")

    def train_one_epoch(self, dataloader, epoch):
        self.model.train()
        scaler = torch.amp.GradScaler(device=self.device)  # Skaler do FP16
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch}", leave=True)
        for data, target in progress_bar:
            data = data.to(self.device).float()
            target = target.float().unsqueeze(1).to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):  # Autocast dla FP16
                output = self.model(data)
                output = torch.sigmoid(output)
                loss = self.criterion(output, target)

            # outputs = self.model(inputs)
            # outputs = torch.sigmoid(outputs)
            # loss = self.criterion(outputs, targets)
            # loss.backward()
            # self.optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            running_loss += loss.item() * data.size(0)
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
        for epoch in range(self.start_epoch, epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.save_top_checkpoints(epoch + 1, val_loss)
            print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

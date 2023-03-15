import torch
import numpy as np
import json
import os


class Training:

    def __init__(self, model, dataloader_train, dataloader_val, steps_train: int, steps_val: int, epochs: int, optimizer,
                 criterion, device, model_dir: str, checkpoint_frequency: int, get_map = None):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.steps_train = steps_train
        self.steps_val = steps_val
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        self.loss = {'train': [], 'val': []}
        self.map_values = {'train': [], 'val': []}
        self.model_dir = model_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.get_map = get_map

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            if epoch % self.checkpoint_frequency == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        losses = []
        map_values = []

        for i, (images, labels) in enumerate(self.dataloader_train):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            loss.backward()
            losses.append(loss.item())

            self.optimizer.step()

            map_value = self.get_map(pred=outputs, true=labels)
            map_values.append(map_value)

            if i == self.steps_train:
                break

        epoch_loss = np.mean(losses)
        self.loss['train'].append(epoch_loss)
        epoch_map = np.mean(map_values)
        self.map_values['train'].append(epoch_map)

        print(f"Epoch {epoch} / {self.epochs}. Train. Loss: {epoch_loss:.3f}. mAP: {epoch_map:.3f}")

    def _validate_epoch(self, epoch):
        self.model.eval()
        losses = []
        map_values = []

        with torch.no_grad():
            for i, data in enumerate(self.dataloader_val, 1):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                losses.append(loss.item())
                map_value = self.get_map(pred=outputs, true=labels)
                map_values.append(map_value)

                if i == self.steps_val:
                    break

        epoch_loss = np.mean(losses)
        self.loss['val'].append(epoch_loss)
        epoch_map = np.mean(map_values)
        self.map_values['val'].append(epoch_map)

        print(f"Epoch {epoch} / {self.epochs}. Validation. Loss: {epoch_loss:.3f}. mAP: {epoch_map:.3f}")

    def _save_checkpoint(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = f'checkpoint_{str(epoch).zfill(3)}.pt'
        model_path = os.path.join(self.model_dir, model_path)
        torch.save(self.model, model_path)

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, 'model.pt')
        torch.save(self.model, model_path)

    def save_loss(self):
        loss_path = os.path.join(self.model_dir, 'loss.json')
        with open(loss_path, 'w') as file:
            json.dump(self.loss, file)

    def save_map(self):
        map_path = os.path.join(self.model_dir, 'map.json')
        with open(map_path, 'w') as file:
            json.dump(self.map_values, file)

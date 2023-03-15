import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import os
from functools import partial
import argparse
import sys

sys.path.append('.')

from yolo.src.dataset import VOCDataset
from yolo.src.loss import YOLOLoss
from yolo.src.model import YOLO
from yolo.src.metrics import get_map
from yolo.src import Training
from yolo.src.utils import load_yaml


def main(config: str):
    config = load_yaml(config)

    transform = transforms.Compose([
        transforms.Resize((config['image']['width'], config['image']['height'])),
        transforms.ToTensor()
    ])

    data_dir = config['data']['directory']
    dataset_train = VOCDataset(
        annotations=os.path.join(data_dir, config['data']['train']),
        images_dir=os.path.join(data_dir, config['data']['images']),
        labels_dir=os.path.join(data_dir, config['data']['labels']),
        transform=transform
    )

    dataset_test = VOCDataset(
        annotations=os.path.join(data_dir, config['data']['val']),
        images_dir=os.path.join(data_dir, config['data']['images']),
        labels_dir=os.path.join(data_dir, config['data']['labels']),
        transform=transform
    )

    S = config['S']
    C = config['C']
    B = config['B']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=config['batch_size'], shuffle=True)
    dataloader_val = DataLoader(dataset=dataset_test, batch_size=config['batch_size'], shuffle=True)

    criterion = YOLOLoss(S, B, C, lambda_coord=config['lambda_coord'], lambda_noobj=config['lambda_noobj'],
                         eps=config['eps'], format=config['image']['format'])

    model = YOLO(in_channels=config['image']['in_channels'], grid_size=S, num_boxes=B, num_classes=C)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    trainer = Training(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        epochs=config['epochs'],
        steps_train=config['steps_train'],
        steps_val=config['steps_val'],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_dir=config['model_directory'],
        checkpoint_frequency=config['checkpoint_frequency'],
        get_map=partial(get_map, S=S, C=C, B=B, format=config['image']['format'])
    )

    trainer.train()

    trainer.save_model()
    trainer.save_loss()
    trainer.save_map()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='yolo/config.yaml', help='Path to the config file')
    args = parser.parse_args()

    main(args.config)

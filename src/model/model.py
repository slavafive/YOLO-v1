import torch.nn as nn


class CNNBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, k: int, s: int, p: int):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.conv(self.relu(x))


class YOLO(nn.Module):

    def __init__(self, in_channels: int, grid_size: int = 7, num_boxes: int = 2, num_classes: int = 20):
        super(YOLO, self).__init__()
        self._create_cnn_layers(in_channels)
        self._create_fc_layers(grid_size, num_boxes, num_classes)

    def forward(self, x):
        return self.fc_layers(self.cnn_layers(x))

    def _create_cnn_layers(self, in_channels: int):
        cnn_layers = [
            CNNBlock(in_channels=in_channels, out_channels=64, k=7, s=2, p=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(in_channels=64, out_channels=192, k=3, s=1, p=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(in_channels=192, out_channels=128, k=1, s=1, p=0),
            CNNBlock(in_channels=128, out_channels=256, k=3, s=1, p=1),
            CNNBlock(in_channels=256, out_channels=512, k=1, s=1, p=0),
            CNNBlock(in_channels=512, out_channels=256, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        for i in range(3):
            cnn_layers += [
                CNNBlock(in_channels=256, out_channels=512, k=1, s=1, p=0),
                CNNBlock(in_channels=512, out_channels=256, k=3, s=1, p=1)
            ]
        cnn_layers += [
            CNNBlock(in_channels=256, out_channels=512, k=1, s=1, p=0),
            CNNBlock(in_channels=512, out_channels=512, k=3, s=1, p=1),
            CNNBlock(in_channels=512, out_channels=1024, k=1, s=1, p=0),
            CNNBlock(in_channels=1024, out_channels=512, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(in_channels=512, out_channels=1024, k=1, s=1, p=0),
            CNNBlock(in_channels=1024, out_channels=512, k=3, s=1, p=1),
            CNNBlock(in_channels=512, out_channels=1024, k=3, s=1, p=1),
            CNNBlock(in_channels=1024, out_channels=1024, k=3, s=1, p=1),
            CNNBlock(in_channels=1024, out_channels=1024, k=3, s=1, p=1),
            CNNBlock(in_channels=1024, out_channels=1024, k=3, s=2, p=1),
            CNNBlock(in_channels=1024, out_channels=1024, k=3, s=1, p=1),
            CNNBlock(in_channels=1024, out_channels=1024, k=3, s=1, p=1)
        ]
        self.cnn_layers = nn.Sequential(*cnn_layers)

    def _create_fc_layers(self, grid_size: int = 7, num_boxes: int = 2, num_classes: int = 20):
        S, B, C = grid_size, num_boxes, num_classes
        fc_layers = [
            nn.Flatten(),
            nn.Linear(in_features=1024 * S * S, out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=S * S * (C + 5 * B))
        ]
        self.fc_layers = nn.Sequential(*fc_layers)

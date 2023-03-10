import torch
import sys

sys.path.append('.')


from yolo.src.model import YOLO


def test_output_dim():
    batch_size = 8
    image_size = 448
    in_channels = 3
    grid_size = 7
    num_boxes = 2
    num_classes = 20

    batch = torch.randn(batch_size, in_channels, image_size, image_size)
    cnn = YOLO(in_channels=in_channels, grid_size=7, num_boxes=2, num_classes=20)
    x = cnn(batch)
    out_dim = grid_size * grid_size * (num_classes + 5 * num_boxes)
    assert x.shape == (batch_size, out_dim)

import torch
import sys
sys.path.append('.')


from yolo.src.metrics import iou


pred = torch.tensor([
    [1, 3, 4, 5],
    [2, 2, 7, 7],
    [-2, -5, 3, 0],
    [1, 2, 5, 4]
])

true = torch.tensor([
    [2, 1, 6, 4],
    [6, 6, 10, 10],
    [-2, -5, 3, 0],
    [1, 2, -5, 4]
])

iou_true = torch.tensor([
    2 / 16,
    1 / 40,
    1,
    0
])

eps = 1e-3


def test_iou():
    iou_pred = iou(pred, true, format='pascal_voc')
    assert (torch.abs(iou_pred - iou_true) < eps).all()

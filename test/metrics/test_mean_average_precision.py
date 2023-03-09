import torch
import sys
sys.path.append('.')


from yolo.src.metrics import mean_average_precision


all_boxes = [
    {
        'pred': [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        'true': [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        'iou_threshold': 0.5,
        'format': 'midpoint',
        'num_classes': 1,
        'map': 1
    },
    {
        'pred': [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        'true': [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        'iou_threshold': 0.5,
        'format': 'midpoint',
        'num_classes': 1,
        'map': 1
    },
    {
        'pred': [
            [0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 1, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        'true': [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        'iou_threshold': 0.5,
        'format': 'midpoint',
        'num_classes': 2,
        'map': 0
    },
    {
        'pred': [
            [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        'true': [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2]
        ],
        'iou_threshold': 0.5,
        'format': 'midpoint',
        'num_classes': 1,
        'map': 5 / 18
    },
]

eps = 1e-4


def test_iou():
    for i, box in enumerate(all_boxes):
        map_ = mean_average_precision(box['pred'], box['true'], box['iou_threshold'], box['format'], box['num_classes'])
        assert abs(map_ - box['map']) < eps

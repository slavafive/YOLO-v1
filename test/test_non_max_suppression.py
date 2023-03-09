import sys

sys.path.append('.')


from yolo.src import non_max_suppression


all_boxes = [
    {
        'before': [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [1, 0.8, 0.5, 0.5, 0.2, 0.4],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1]
        ],
        'after': [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1]
        ],
        'prob_threshold': 0.2,
        'iou_threshold': 7 / 20,
        'format': 'midpoint'
    },
    {
        'before': [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1]
        ],
        'after': [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1]
        ],
        'prob_threshold': 0.2,
        'iou_threshold': 7 / 20,
        'format': 'midpoint'
    },
    {
        'before': [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [2, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1]
        ],
        'after': [
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [2, 0.8, 0.25, 0.35, 0.3, 0.1]
        ],
        'prob_threshold': 0.2,
        'iou_threshold': 7 / 20,
        'format': 'midpoint'
    },
    {
        'before': [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1]
        ],
        'after': [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1]
        ],
        'prob_threshold': 0.2,
        'iou_threshold': 9 / 20,
        'format': 'midpoint'
    }
]


def test_non_max_suppression():
    for boxes in all_boxes:
        result = non_max_suppression(boxes['before'], boxes['prob_threshold'], boxes['iou_threshold'], boxes['format'])
        assert sorted(result) == sorted(boxes['after'])

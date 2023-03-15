import torch
from collections import defaultdict

from yolo.src.metrics import iou
from yolo.src.utils import flatten


def non_max_suppresion_for_one_class(boxes: list, prob_threshold: float, iou_threshold: float, format: str):
    boxes = [box for box in boxes if box[1] > prob_threshold]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    new_boxes = []
    while len(boxes) > 0:
        chosen_box = boxes.pop(0)
        new_boxes.append(chosen_box)
        boxes = [box for box in boxes if (iou(torch.tensor(box[2:]).unsqueeze(0),
                            torch.tensor(chosen_box[2:]).unsqueeze(0), format) < iou_threshold).item()]
    return new_boxes


def non_max_suppression(pred: list, prob_threshold: float, iou_threshold: float, format: str = 'pascal_voc'):
    """
    pred: [class_num, score, x1, y1, x2, y2]
    """
    boxes_dict = defaultdict(lambda: [])
    for box in pred:
        boxes_dict[box[0]].append(box)
    for class_num, class_boxes in boxes_dict.items():
        boxes_dict[class_num] = non_max_suppresion_for_one_class(class_boxes, prob_threshold, iou_threshold, format)
    return flatten(boxes_dict.values())

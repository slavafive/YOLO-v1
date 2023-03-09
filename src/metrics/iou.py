import torch


def area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def iou(pred, true, format: str = 'pascal_voc'):

    x_min = torch.maximum(pred[:, 0], true[:, 0]).unsqueeze(1)
    y_min = torch.maximum(pred[:, 1], true[:, 1]).unsqueeze(1)

    if format == 'pascal_voc':

        x_max = torch.minimum(pred[:, 2], true[:, 2]).unsqueeze(1)
        y_max = torch.minimum(pred[:, 3], true[:, 3]).unsqueeze(1)

    elif format == 'coco':

        x_max = torch.minimum(pred[:, 0] + pred[:, 2], true[:, 0] + true[:, 2]).unsqueeze(1)
        y_max = torch.minimum(pred[:, 1] + pred[:, 3], true[:, 1] + true[:, 3]).unsqueeze(1)

    boxes = torch.concat([x_min, y_min, x_max, y_max], dim=1)

    intersection = area(boxes)
    union = area(pred) + area(true) - intersection + 1e-6
    return intersection / union

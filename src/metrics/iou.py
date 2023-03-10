import torch


def area(boxes):
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def midpoint_to_pascal_voc(tensor):
    return torch.concat([
        tensor[..., 0:1] - tensor[..., 2:3] / 2,
        tensor[..., 1:2] - tensor[..., 3:4] / 2,
        tensor[..., 0:1] + tensor[..., 2:3] / 2,
        tensor[..., 1:2] + tensor[..., 3:4] / 2
    ], dim=1)


def iou(pred, true, format: str = 'pascal_voc'):

    if format == 'midpoint':
        pred = midpoint_to_pascal_voc(pred)
        true = midpoint_to_pascal_voc(true)
        format = 'pascal_voc'

    x_min = torch.maximum(pred[..., 0:1], true[..., 0:1])
    y_min = torch.maximum(pred[..., 1:2], true[..., 1:2])

    if format == 'pascal_voc':

        x_max = torch.minimum(pred[..., 2:3], true[..., 2:3])
        y_max = torch.minimum(pred[..., 3:4], true[..., 3:4])

    elif format == 'coco':

        x_max = torch.minimum(pred[..., 0:1] + pred[..., 2:3], true[..., 0:1] + true[..., 2:3])
        y_max = torch.minimum(pred[..., 1:2] + pred[..., 3:4], true[..., 1:2] + true[..., 3:4])

    else:
        raise ValueError(f"""Incorrect bounding box format. Possible values: ['pascal_voc', 'coco', 'midpoint'].
                        Got {format} instead.""")

    boxes = torch.concat([x_min, y_min, x_max, y_max], dim=-1)

    intersection = area(boxes)
    union = area(pred) + area(true) - intersection + 1e-6
    return intersection / union

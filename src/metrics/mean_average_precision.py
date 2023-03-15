import torch
import numpy as np

from yolo.src.non_max_suppression import non_max_suppression
from yolo.src.metrics import iou


def mean_average_precision(pred, true, iou_threshold: float, format: str = 'pascal_voc', num_classes: int = 20):
    """
    pred: [image_num, class_num, score, x1, y1, x2, y2]
    """
    auc_values = []
    for class_num in range(num_classes):
        boxes = [box for box in pred if box[1] == class_num]
        boxes = sorted(boxes, key=lambda x: x[2], reverse=True) # sort by score
        TP = FP = 0
        precision = [1]
        recall = [0]
        for pred_box in boxes:
            image_num = pred_box[0]
            gt_boxes = [box for box in true if box[0] == image_num and box[1] == class_num]
            max_iou, corresponding_box_index = 0, -1
            for i, gt_box in enumerate(gt_boxes):
                current_iou = iou(torch.tensor(pred_box[3:]).unsqueeze(0), torch.tensor(gt_box[3:]).unsqueeze(0), format)[0].item()
                if current_iou > max_iou:
                    max_iou = current_iou
                    corresponding_box_index = i
            if corresponding_box_index == -1 or max_iou < iou_threshold:
                FP += 1
            else:
                TP += 1
                gt_boxes.pop(corresponding_box_index)
            precision.append(TP / (TP + FP))
            recall.append(TP / len(boxes))
        auc_values.append(torch.trapz(torch.tensor(precision), torch.tensor(recall)).item())
    return np.mean(auc_values)


def get_map(pred, true, S: int = 7, C: int = 20, B: int = 2, format: str = 'pascal_voc'):
    pred = pred.reshape(-1, S, S, C + 5 * B)
    true = true.reshape(-1, S, S, C + 5)

    iou_1 = iou(pred[..., 21:25], true[..., 21:25], format)
    iou_2 = iou(pred[..., 26:30], true[..., 21:25], format)

    ious = torch.concat([iou_1.unsqueeze(-1), iou_2.unsqueeze(-1)], dim=-1)
    _, indices = torch.max(ious, -1)

    exists_box = true[..., 20] # either 1 or 0
    exists_box_unsqueezed = exists_box.unsqueeze(3).repeat(1, 1, 1, 4)
    indices_unsqueezed = indices.unsqueeze(3).repeat(1, 1, 1, 4)
    pred_box = exists_box_unsqueezed * (
            indices_unsqueezed * pred[..., 26:30] + (1 - indices_unsqueezed) * pred[..., 21:25]
    )
    true_box = exists_box_unsqueezed * true[..., 21:25]
    pred_conf, pred_class = torch.max(pred[..., 0:20], dim=-1)
    true_conf = exists_box
    true_class = true[..., 20]

    true_boxes = []
    for i in torch.argwhere(true_conf).tolist():
        true_boxes.append([
            i[0], int(true_class[i[0]][i[1]][i[2]].item()), true_conf[i[0]][i[1]][i[2]].item(), *[coord.item() for coord in true_box[i[0]][i[1]][i[2]]]
        ])

    pred_boxes = []
    for image_num, image in enumerate(pred_box):
        image_pred_boxes = []
        for i in range(len(image)):
            for j in range(len(image[i])):
                image_pred_boxes.append([
                    int(pred_class[image_num][i][j].item()), pred_conf[image_num][i][j].item(),
                    *[coord.item() for coord in pred_box[image_num][i][j]]
                ])
        image_pred_boxes = non_max_suppression(image_pred_boxes, prob_threshold=0.3, iou_threshold=0.5, format=format)
        pred_boxes += [[image_num, *box] for box in image_pred_boxes]
    return mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, format=format, num_classes=20)

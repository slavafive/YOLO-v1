import torch
import numpy as np


from yolo.src.metrics import iou


def mean_average_precision(pred, true, iou_threshold, format: str = 'pascal_voc', num_classes: int = 20):
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


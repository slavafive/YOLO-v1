import torch
import torch.nn as nn
from yolo.src.metrics import iou


class YOLOLoss(nn.Module):

    def __init__(self, S: int = 7, B: int = 2, C: int = 20, lambda_coord: float = 5, lambda_noobj: float = 0.5,
                 eps: float = 1e-6):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.eps = eps

    def _reshape(self, tensor):
        return tensor.reshape(-1, self.S, self.S, self.C + 5 * self.B)

    def forward(self, pred, true, format: str = 'pascal_voc'):
        # pred: [p1, p2, ..., p20, c1, x1, y1, w1, h1, c2, x2, y2, w2, h2] - 30 values
        # target: [p1, p2, ..., p20, c1, x1, y1, w1, h1] - 25 values
        pred = pred.reshape(-1, self.S, self.S, self.C + 5 * self.B)
        true = true.reshape(-1, self.S, self.S, self.C + 5)

        iou_1 = iou(pred[..., 21:25], true[..., 21:25], format)
        iou_2 = iou(pred[..., 26:30], true[..., 21:25], format)

        ious = torch.concat([iou_1.unsqueeze(-1), iou_2.unsqueeze(-1)], dim=-1)
        _, indices = torch.max(ious, -1)

        exists_box = true[..., 20]
        exists_box_unsqueezed = exists_box.unsqueeze(3).repeat(1, 1, 1, 4)
        indices_unsqueezed = indices.unsqueeze(3).repeat(1, 1, 1, 4)
        pred_box = exists_box_unsqueezed * (
                indices_unsqueezed * pred[..., 21:25] + (1 - indices_unsqueezed) * pred[..., 26:30]
        )
        true_box = exists_box_unsqueezed * true[..., 21:25]

        true_box[..., 2:4] = true_box[..., 2:4].clone().abs().sqrt()
        pred_box[..., 2:4] = pred_box[..., 2:4].clone().sign() * (pred_box[..., 2:4].clone().abs() + self.eps).sqrt()

        return self.mse(torch.flatten(true_box, end_dim=-2), torch.flatten(pred_box, end_dim=-2))

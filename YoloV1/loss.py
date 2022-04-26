import torch
import torch.nn as nn
from iou import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=2):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        bbox1_idx, bbox1_idx_end = self.C + 1, self.C + 5
        bbox2_idx, bbox2_idx_end = bbox1_idx + 5, bbox1_idx_end + 5
        iou_b1 = intersection_over_union(
            predictions[..., bbox1_idx : bbox1_idx_end],
            target[..., bbox1_idx : bbox1_idx_end]
        )
        iou_b2 = intersection_over_union(
            predictions[..., bbox2_idx : bbox2_idx_end],
            target[..., bbox1_idx : bbox1_idx_end]
        )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., bbox1_idx - 1].unsqueeze(3)  # Iobj_i

        ############################
        ### for box coordinates  ###
        ############################
        box_predictions = exists_box * (
            (
                best_box * predictions[..., bbox2_idx : bbox2_idx_end]
                + (1 -best_box) * predictions[..., bbox1_idx : bbox1_idx_end]
            )
        )

        box_targets = exists_box * target[..., bbox1_idx : bbox1_idx_end]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt( torch.abs( box_predictions[..., 2:4] + 1e-6) )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        #(N, S, S, 3) => (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
            )

        #######################
        ### for object loss ###
        #######################
        pred_box = (
            best_box * predictions[..., bbox2_idx - 1 : bbox2_idx]
            + (1 - best_box) * predictions[..., bbox1_idx - 1 : bbox1_idx]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., bbox1_idx - 1 : bbox1_idx])
        )

        ##########################
        ### for no object loss ###
        ##########################

        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., bbox1_idx - 1 : bbox1_idx], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., bbox1_idx - 1 : bbox1_idx], start_dim= 1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., bbox2_idx - 1: bbox2_idx], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., bbox1_idx - 1: bbox1_idx], start_dim=1)
        )

        ######################
        ### for class loss ###
        ######################

        # (N, S, S, 2) -> (N*S*S, 2)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=2)
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss


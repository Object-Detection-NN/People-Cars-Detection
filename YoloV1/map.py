import torch
from collections import Counter
from iou import intersection_over_union

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="top_left", num_classes=2):
    # pred_boxes = [ [train_idx, class_pred, prob_score, x1, y1, w, h], ... ]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # image 0 has 3 bboxes
        # image 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter( [gt[0] for gt in ground_truths] )

        # amount_bboxes = {0: tensor([0, 0, 0]), 1: tensor([0, 0, 0, 0, 0])}
        for key, values in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(values)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_boxes = len(ground_truths)

        if total_true_boxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truths_img = list(filter(lambda box: box[0] == detection[0], ground_truths))

            best_iou = 0

            for idx, gt in enumerate(ground_truths_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),
                                              torch.tensor(gt[3:]),
                                              box_format=box_format
                                              )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

if __name__ == "__main__":

    t1_preds = [
        [0, 0, 0.9, 0.4, 0.1, 0.3, 0.2],
        [0, 0, 0.8, 0.2, 0.5, 0.3, 0.2],
        [0, 0, 0.7, 0.7, 0.6, 0.2, 0.2],
    ]
    t1_targets = [
        [0, 0, 0.9, 0.4, 0.1, 0.3, 0.2],
        [0, 0, 0.8, 0.2, 0.5, 0.3, 0.2],
        [0, 0, 0.7, 0.7, 0.6, 0.2, 0.2],
    ]
    t1_correct_mAP = 1

    epsilon = 1e-4
    mean_avg_prec = mean_average_precision(
        t1_preds,
        t1_targets,
        iou_threshold=0.5,
        box_format="top_left",
        num_classes=1,
    )
    assert(abs(t1_correct_mAP - mean_avg_prec) < epsilon)


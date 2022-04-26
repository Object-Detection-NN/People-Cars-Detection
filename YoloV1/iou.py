import torch

def intersection_over_union(boxes_pred, boxes_labels, box_format="top_left"):
    # boxes_pred shape: (N, 4), N -> number of bboxes

    if box_format == "top_left":
        box1_x1 = boxes_pred[..., 0:1]
        box1_y1 = boxes_pred[..., 1:2]
        box1_x2 = box1_x1 + boxes_pred[..., 2:3]
        box1_y2 = box1_y1 + boxes_pred[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = box2_x1 + boxes_labels[..., 2:3]
        box2_y2 = box2_y1 + boxes_labels[..., 3:4]

    if box_format == "corners":
        box1_x1 = boxes_pred[..., 0:1]
        box1_y1 = boxes_pred[..., 1:2]
        box1_x2 = boxes_pred[..., 2:3]
        box1_y2 = boxes_pred[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) is for not intersecting boxes
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x1 - box1_x2) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x1 - box2_x2) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


if __name__ == "__main__":
    # test cases we want to run
    # corners
    t1_box1 = torch.tensor([2, 2, 6, 6])
    t1_box2 = torch.tensor([4, 4, 7, 8])
    t1_correct_iou = 4 / 24
    # top_left
    t2_box1 = torch.tensor([2, 2, 6, 6])
    t2_box2 = torch.tensor([4, 4, 7, 8])
    t2_correct_iou = 16 / 76

    epsilon = 0.001
    iou = intersection_over_union(t1_box1, t1_box2, box_format="corners")
    assert(torch.abs(iou - t1_correct_iou) < epsilon)
    iou = intersection_over_union(t2_box1, t2_box2, box_format="top_left")
    assert(torch.abs(iou - t2_correct_iou) < epsilon)

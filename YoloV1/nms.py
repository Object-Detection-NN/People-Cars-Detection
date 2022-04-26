import torch
from iou import intersection_over_union

def non_max_suppression(bboxes, iou_threshold, probability_threshold, box_format="top_left"):
    # bboxes = [ [class, probability, x1, y1, w, h], ... ]
    assert type(bboxes) == list
    bboxes = list(filter(lambda box: box[1] > probability_threshold, bboxes))
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = list(filter(lambda box: box[0] != chosen_box[0]
                                or intersection_over_union(
                                        torch.tensor(chosen_box[2:]),
                                        torch.tensor(box[2:]),
                                        box_format=box_format
                                )
                                < iou_threshold
                        ,bboxes))

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

if __name__ == "__main__":
    t1_boxes = [
        [1, 1, 0.3, 0.2, 0.4, 0.5],
        [1, 0.8, 0.4, 0.3, 0.2, 0.4],
        [1, 0.7, 0.1, 0.3, 0.3, 0.1],
        [1, 0.05, 0.05, 0.05, 0.1, 0.1],
    ]

    c1_boxes = [[1, 1, 0.3, 0.2, 0.4, 0.5], [1, 0.7, 0.1, 0.3, 0.3, 0.1]]

    bboxes = non_max_suppression(
        t1_boxes,
        probability_threshold=0.2,
        iou_threshold= 7 / 20,
        box_format="top_left",
    )
    assert(sorted(bboxes) == sorted(c1_boxes))

import torch
import fiftyone as fo
from PIL import Image
import os


class COCODataset(torch.utils.data.Dataset):
    classes_str = {
        "car": 0,
        "person": 1,
    }
    classes_int = {
        0: "car",
        1: "person"
    }

    def __init__(self, fo_dataset_name, S=7, B=2, C=2, transform=None):
        self.annotations = fo.load_dataset(fo_dataset_name)
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        boxes = []
        for detection in self.annotations[item:item+1].first().ground_truth.detections:
            boxes.append([COCODataset.classes_str[detection.label], *detection.bounding_box])

        image_path = self.annotations[item:item+1].first().filepath
        image = Image.open(image_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


if __name__ == "__main__":
    dataset = COCODataset("train_filtered_50")
    print(len(dataset))
    print(dataset[0])

import fiftyone as fo
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from nn_model import Yolo_v1
from map import mean_average_precision
from nms import non_max_suppression
from loss import YoloLoss
from dataset import COCODataset
from bboxes_utils import get_bboxes, cellboxes_to_boxes
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


seed = 123
torch.manual_seed(seed)

learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
batch_size = 16
weight_decay = 0
epochs = 100
num_workers = 1
pin_memory = True
load_model = True
train_model = True
load_model_file = "model2k.pth"
out_model_file = load_model_file
run_fifty_one = True
train_dataset_name = "train_filtered_2k" # "train_filtered_50" #


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    return sum(mean_loss)/len(mean_loss)

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



def main():
    model = Yolo_v1(split_size=7, num_boxes=2, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = YoloLoss()

    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optimizer)

    train_dataset = COCODataset(train_dataset_name, transform=transform)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )



    if train_model:
        map_y = []
        mean_loss = []
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")
            pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
            mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="top_left")
            map_y.append(float(mean_avg_prec))
            print(f"Train mAP: {mean_avg_prec}")
            mean_loss.append( train_fn(train_loader, model, optimizer, loss_fn) )

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint, filename=out_model_file)
        print(f"loss: {mean_loss}\nepochs: {epochs}")
        print(f"map: {map_y}")

        plt.plot(list(range(1, 1 + epochs)), map_y)
        plt.title(f"Dataset: {train_dataset_name} mean average precision")
        plt.xlabel("Epoch")
        plt.show()

        plt.plot(list(range(1, 1 + epochs)), mean_loss)
        plt.title(f"Dataset: {train_dataset_name} mean loss")
        plt.xlabel("Epoch")
        plt.show()


    if run_fifty_one:
        predictions_dataset = fo.load_dataset(train_dataset_name)

        for sample, (image, _) in zip(predictions_dataset, train_dataset):
            detections = []
            image = image.to(device)
            bboxes = cellboxes_to_boxes(model(image.unsqueeze(0)))
            bboxes = non_max_suppression(*bboxes, iou_threshold=0.5, probability_threshold=0.4, box_format="top_left")
            for box in bboxes:
                detections.append(fo.Detection(bounding_box=box[2:6], label=COCODataset.classes_int[int(box[0])], confidence=box[1]))
            sample["predictions"] = fo.Detections(detections=detections)
            sample.save()

        session = fo.launch_app(predictions_dataset)
        session.wait()


if __name__ == "__main__":
    main()

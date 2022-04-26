import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.image as img
import fiftyone.utils.torch as fot


dataset_to_download_name = "coco-2017"
save_downloaded_dataset = False
downloaded_dataset_name = "coco-2017-train-car-person"
classes = ["car", "person"]
split = "train"
label_types = "detections"
save_filtered_dataset = True
filtered_dataset_name = "train_filtered_20k"
max_samples = 20000


def download_and_filter_dataset():
    # Load the COCO-2017 validation split into a FiftyOne dataset
    # This will download the dataset from the web, if necessary
    dataset = foz.load_zoo_dataset(
        dataset_to_download_name,
        split=split,
        label_types=label_types,
        classes=classes,
        max_samples=max_samples,
    )

    # Give the dataset a new name, and make it persistent so that you can
    # work with it in future sessions
    dataset.name = downloaded_dataset_name
    dataset.persistent = save_downloaded_dataset

    ### print some data ###
    # dataset.compute_metadata()
    # print(dataset)
    # print(dataset.first().metadata)
    # print(dataset.first().ground_truth.detections)

    # for data in dataset:
    #     print("\n==========================")
    #     print("id    width    height")
    #     print(f"{data.id} | {data.metadata.width} | {data.metadata.height}")
    #     print("*** detections ***")
    #     print("id    label   is_crowd  bounding_box")
    #     for det in data.ground_truth.detections:
    #         print("--------------------------------------")
    #         print(f"{det.id} | {det.label} | {det.iscrowd} | {det.bounding_box}")

    ### filter dataset ###
    # Bboxes are in [top-left-x, top-left-y, width, height] format
    bbox_area = F("bounding_box")[2] * F("bounding_box")[3]
    searched_label = F("label").is_in(classes)

    # Only contains images whose bounding boxes area is at least 10% of the image
    view = dataset.match(F("ground_truth.detections").filter((bbox_area >= 0.1) & searched_label).length() > 0)

    # remove other class bboxes
    for sample in dataset:
        to_be_removed = []
        for det in sample.ground_truth.detections:
            if det.label not in classes:
                to_be_removed.append(det)
        for tbr in to_be_removed:
            sample.ground_truth.detections.remove(tbr)
        sample.save()


    # create and save new filtered dataset
    dataset_filtered = fo.Dataset(filtered_dataset_name)
    dataset_filtered.add_samples(view)
    dataset_filtered.persistent = save_downloaded_dataset


def resize_images(dataset_name, size=(448, 448)):
    dataset = fo.load_dataset(dataset_name)
    img.transform_images(sample_collection=dataset, size=size)
    dataset.compute_metadata(overwrite=True)


if __name__ == '__main__':
    # download and filter dataset if dataset does not exist
    if not fo.dataset_exists(filtered_dataset_name):
        download_and_filter_dataset()
        resize_images(filtered_dataset_name)

    dataset_filtered = fo.load_dataset(filtered_dataset_name)

    # Visualize the dataset in the App
    session = fo.launch_app(dataset_filtered)
    session.wait()

import matplotlib.pyplot as plt
import json

CLASSES = ["bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"]


def data_exploration(label_path, image_size):
    """
        The goal is to get an idea of what's in the dataset, how many parts of an image correspond to a label, etc...
    """
    # Load labels
    labels_file = open(label_path, "r")
    labels = json.load(labels_file)
    labels_file.close()
    labels_sample_sizes = [0 for _ in range(len(CLASSES))]

    # Sorting usable image from the dataset (not selecting the 
    for i in range(len(labels)):
        for j in labels[i]["labels"]:
            # excluding the drivable area and lane part from the selection
            if j["category"] in ["drivable area", "lane"]:
                continue

            x1, y1, x2, y2 = (
                round(float(j["box2d"]["x1"])),
                round(float(j["box2d"]["y1"])),
                round(float(j["box2d"]["x2"])),
                round(float(j["box2d"]["y2"])))
            # excluding the image that are of too low quality
            if (x2 - x1) * (y2 - y1) < image_size:
                continue

            labels_sample_sizes[CLASSES.index(j["category"])] += 1

    # Display training samples sizes
    plt.subplots(figsize=(len(CLASSES), 5), dpi=100)
    plt.bar(CLASSES, height=labels_sample_sizes)
    plt.title("Distribution of the training set according to label")
    plt.show()
    print(CLASSES)
    print(labels_sample_sizes)


if __name__ == "__main__":
    data_exploration("./dataset/train/train_labels.json", 500)

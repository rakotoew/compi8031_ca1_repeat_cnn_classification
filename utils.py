import os
import cv2

from main import MIN_PIXEL_IMAGE


def resize_image(image, resolution):
    """
    function to resize an image to the desired resolution
    :param image: the image wanted to be resized
    :param resolution: the desired resolution
    :return: image
    """
    return cv2.resize(image, (resolution, resolution))


def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_writing(output_directory, image):
    """
    write a specified image to a specified directory
    :return: boolean
    """
    if sample_sizes[CLASSES.index(l["category"])] < sample_size_limit:
        class_directory = os.path.join(output_directory, l["category"])
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)

        cv2.imwrite(os.path.join(class_directory, str(l["id"]) + ".jpg"), image)
        sample_sizes[CLASSES.index(l["category"])] += 1
        return True
    return False


def image_extraction(dataset_path, label, size_image, output_directory):
    """
    Extract part of image according to label from a dataset to a specified directory
    :param dataset_path:
    :param label:
    :param size_image:
    :param output_directory:
    :return:
    """
    image = cv2.imread(os.path.join(dataset_path, label["name"]))
    extracted_part_count = 0
    excluded_part_count = 0

    for j in label["labels"]:
        # excluding the drivable area and lane part from the selection
        if j["category"] in ["drivable area", "lane"]:
            continue

        x1, y1, x2, y2 = (
            round(float(j["box2d"]["x1"])),
            round(float(j["box2d"]["y1"])),
            round(float(j["box2d"]["x2"])),
            round(float(j["box2d"]["y2"])))
        # excluding the image that are of too low quality
        if (x2 - x1) * (y2 - y1) < MIN_PIXEL_IMAGE:
            excluded_part_count += 1
            continue

        extracted_image = image[y1:y2, x1:x2]
        extracted_image = resize_image(extracted_image, 100)
        extracted_image = gray_scale(extracted_image)

        if not image_writing(j, output_directory, extracted_image, sample_sizes, samples_size_limit):
            excluded_part_count += 1
            continue
        extracted_part_count += 1
    return excluded_part_count, extracted_part_count

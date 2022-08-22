import os
import cv2

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
    if 0:
        return True
    else:
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
        if (x2 - x1) * (y2 - y1) < size_image:
            excluded_part_count += 1
            continue

        extracted_image = image[y1:y2, x1:x2]
        extracted_image = resize_image(extracted_image)
        extracted_image = gray_scale(extracted_image)

        if not image_writing(output_directory, extracted_image):
            excluded_part_count += 1
            continue
        extracted_part_count += 1
    return excluded_part_count, extracted_part_count

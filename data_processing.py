import json
import os
import cv2
import shutil
import random
from tqdm import tqdm

import utils
from data_exploration import CLASSES


def preprocess(dataset_path, labels_path, output_directory, sample_size_limit):
    # Checking output directory
    if os.path.exists(output_directory):
        print("Overwriting existing output directory...")
        shutil.rmtree(output_directory)
    # Loading labels
    print("Loading labels...")
    labels_file = open(labels_path, "r")
    labels = json.load(labels_file)
    labels_file.close()
    print(f"Labels loaded, found {len(labels)} labels.\n")

    # Extract sub images from each image
    print("Extracting images...")
    extracted_total = 0
    skipped_total = 0
    sample_sizes = [0 for _ in range(len(CLASSES))]
    for i in tqdm(range(len(labels))):
        extracted, skipped = utils.image_extraction(dataset_path, labels[i], output_directory, sample_sizes,
                                                    sample_size_limit)
        extracted_total += extracted
        skipped_total += skipped
    print(f"{extracted_total} images extracted, ({skipped_total} skipped).\n")


def create_test_set(output_directory, test_set_size_percentage):
    """Creates a test set by moving images from the training set"""
    for clazz in CLASSES:
        # Create "output/test/" path
        test_class_directory = os.path.join(output_directory, "test", clazz)
        train_class_directory = os.path.join(output_directory, "train", clazz)
        if not os.path.exists(test_class_directory):
            os.makedirs(test_class_directory)

        # Counting how many images there are in the train class folder
        # to determine how many images will be moved to the test folder.
        nb_img_train_set = len(os.listdir(train_class_directory))
        nb_img = nb_img_train_set * test_set_size_percentage // 100

        # Moving images
        for i in range(nb_img):
            img_path = random.choice(os.listdir(train_class_directory))
            shutil.move(os.path.join(train_class_directory, img_path), os.path.join(test_class_directory, img_path))

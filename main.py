import data_exploration as exploration
import data_processing as process

MIN_PIXEL_IMAGE = 8000
SAMPLE_SIZE_LIMIT = 20000
TEST_SET_SIZE_PERCENTAGE = 20

if __name__ == "__main__":
    # exploring the current dataset
    exploration.data_exploration("./dataset/train/labels.json", MIN_PIXEL_IMAGE)
    # Preprocess training set
    process.preprocess("./dataset/train", "./dataset/train/labels.json", "./dataset/output/train", SAMPLE_SIZE_LIMIT)
    # Preprocess validation set
    process.preprocess("./dataset/val", "./dataset/val/labels.json", "./dataset/output/val", SAMPLE_SIZE_LIMIT)
    process.create_test_set("./dataset/out", TEST_SET_SIZE_PERCENTAGE)
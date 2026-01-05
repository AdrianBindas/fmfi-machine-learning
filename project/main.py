import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
from utils import init_dataframe, Pipeline
import segmentation
import logging
logging.basicConfig(filename='log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set dataset path
TRAIN_DATA = Path("project/data/ISIC2018_Task3_Training_Input")
TRAIN_LABELS = Path("project/data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")

VALID_DATA = Path("project/data/ISIC2018_Task3_Validation_Input")
VALID_LABELS = Path("project/data/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv")

TEST_DATA = Path("project/data/ISIC2018_Task3_Test_Input")
TEST_LABELS = Path("project/data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")

PROCESSED_IMAGES_PATH = Path("project/data/processed/")


if __name__ == "__main__":
    train_df = init_dataframe(TRAIN_DATA, TRAIN_LABELS)
    valid_df = init_dataframe(VALID_DATA, VALID_LABELS)
    test_df = init_dataframe(TEST_DATA, TEST_LABELS)
    pipeline = Pipeline(train_df=train_df, valid_df=valid_df, test_df=test_df, output_path=PROCESSED_IMAGES_PATH)

    pipeline.print_df()

    pipeline.apply_to_image_and_save('image_path', segmentation.apply_clahe, 'clahe', on_split=['test', 'valid', 'train'])

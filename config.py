'''
This file contains all the shared constant values across different files/scripts
Auth: ABx_Lab
'''

# Repo for processed data
GM_DATA_DIR = "training"
PROCESSED_GM_DIR = "data_processed"
OUTPUT_DIR = "data_skeletons"

# Skeleton and training parameters
FRAMES_PER_SEQUENCE = 16
MAX_VIDEOS_PER_CLASS = 500  # Caps massive classes to prevent imbalance
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 34
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
EXPECTED_FEATURES = 225 # contains all the essential skeleton points of a human body
ZERO_FRAME_THRESHOLD = 0.80   # skip sequence if >80% of frames are all-zero
WEIGHT_DECAY      = 1e-4


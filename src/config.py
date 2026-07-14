from pathlib import Path

# --- paths ---
DATA_DIR = Path("data/BUSI")          # expects benign/, malignant/, normal/ subfolders
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
TORCHSCRIPT_MODEL_PATH = CHECKPOINT_DIR / "model_torchscript.pt"

# --- data ---
IMG_SIZE = 256           # resize target; balances detail vs. training speed on CPU/small GPU
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# --- training ---
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

# --- inference ---
MASK_THRESHOLD = 0.5     # sigmoid output above this is classified as tumor

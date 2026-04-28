import os

# Directories
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, "img")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# Prompt

# Image for tool
CURRENT_IMAGE = None

# Model setting
LLM_PATH   = os.path.join(MODEL_DIR, "Qwen3-4B-AWQ") 
GPU_UTIL     = 0.8
MAX_LENGTH   = 8192
MAX_NUM_SEQS = 1
LIMIT_INPUT  = {"video": 0}

SAMPLING_PARAMS  = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 1000
}

STT_PATH = os.path.join(MODEL_DIR, "faster-whisper-small")
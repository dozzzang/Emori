import os

# Data 저장 경로
DATA_DIR = "Emotion_EEG_Code/Data/"

OUTPUT_JSON_FILE = os.path.join(DATA_DIR, "Augmented_Report_Data.json")
BASE_INPUT_FILE = os.path.join(DATA_DIR, "RECORD_20250515__1.txt")
LLAMA3_ADAPTER = os.path.join(DATA_DIR, "Llama3_Result")
ASSISTANT_LABELS = os.path.join(DATA_DIR, "Manual_Assistant_Labels.csv")
TRAIN_JSONL_FILE = os.path.join(DATA_DIR, "Train_Data.jsonl")

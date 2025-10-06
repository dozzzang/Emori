import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# ===================================================================
# 프로젝트 루트 경로 설정 및 constants 모듈 로드
# ===================================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 이 경로는 실행 스크립트의 위치에 따라 조정이 필요할 수 있습니다.
project_root = os.path.join(current_script_dir, "..")

if project_root not in sys.path:
    sys.path.append(project_root)

import constants

# ===================================================================
# 0. 설정 변수 / Llama-3.1-8B-Instruct 모델 사용 (접근권한 및 토큰 필요)
# ===================================================================
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
JSONL_FILE = constants.INPUT_JSONL_FILE
OUTPUT_DIR = constants.LLAMA3_ADAPTER

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"출력 폴더를 생성했습니다: {OUTPUT_DIR}")

# 파라미터 설정
OPTIMAL_EPOCHS = 10
OPTIMAL_BATCH_SIZE = 1
LOGGING_FREQUENCY = 5
ACCUMULATION_STEPS = 8
GROUP_BY_LENGTH_FLAG = False


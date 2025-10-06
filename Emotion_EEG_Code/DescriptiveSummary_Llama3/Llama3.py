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

# ===================================================================
# 1. 모델 및 토크나이저 설정
# ===================================================================

# 4-bit 양자화 설정 (GPU 메모리 절약)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id


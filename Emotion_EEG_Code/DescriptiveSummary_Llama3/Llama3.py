import torch
import os
import sys
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

# ===================================================================
# 2. PEFT (LoRA) 설정 및 모델 변환
# ===================================================================
# LoRA 설정
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
print("Model successfully converted to PEFT (LoRA) model.")


# ===================================================================
# 3. 데이터 로드 및 포매팅
# ===================================================================

# 3.1 데이터셋 로드
print(f"Loading dataset from {JSONL_FILE}...")
try:
    # JSONL 파일을 로드합니다.
    dataset = load_dataset("json", data_files=JSONL_FILE, split="train")
except Exception as e:
    print(
        f"데이터셋 로드 오류: {e}. 'datasets' 라이브러리가 설치되어 있는지 확인하세요."
    )
    exit()


# 3.2 Llama 3.1 Chat Template 포맷 함수 정의
def apply_chat_template_to_text(example):
    """
    'messages' 컬럼의 내용을 Llama 3.1의 채팅 템플릿 문자열로 변환하여 'text' 컬럼에 저장합니다.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# 3.3 데이터셋에 포매팅 적용
dataset = dataset.map(
    apply_chat_template_to_text,
    remove_columns=dataset.column_names,
    desc="Applying chat template and creating 'text' column",
)

# # 데이터셋의 첫 번째 샘플 확인 (테스트)
# print("\n--- Formatted Dataset Example (First 100 characters) ---")
# print(dataset[0]["text"][:100] + "...")
# print("---------------------------------------------------------")


# ===================================================================
# 4. TrainingArguments 및 SFTTrainer 설정
# ===================================================================

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=OPTIMAL_EPOCHS,
    per_device_train_batch_size=OPTIMAL_BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    optim="paged_adamw_8bit",
    save_strategy="no",
    eval_strategy="no",
    logging_steps=LOGGING_FREQUENCY,
    save_total_limit=0,
    learning_rate=3e-4,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    group_by_length=GROUP_BY_LENGTH_FLAG,
    report_to="none",
)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ===================================================================
# 5. 훈련 실행
# ===================================================================
print("Starting training...")
trainer.train()

# 학습 종류 후 pc 끊김 문제 해결 부분
print("Moving adapter to CPU and saving (adapter-only, safetensors)...")

model.eval()
model.to("cpu")  # CPU로 옮겨 저장 과정의 VRAM 압박 완화
import torch

torch.cuda.empty_cache()  # 남은 VRAM 비우기

# PEFT(PeftModel)라면 어댑터만 저장
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Adapter & tokenizer saved to {OUTPUT_DIR}")

# ===================================================================
# 6. 프로그램 종료 및 CUDA 컨텍스트 해제 (최종 안정화)
# ===================================================================

# Python 프로세스가 완전히 종료되도록 명시적인 명령어 추가
try:
    # Python에게 메모리 정리를 요청합니다.
    import gc

    gc.collect()

    # CUDA 메모리를 마지막으로 한 번 더 비웁니다.
    torch.cuda.empty_cache()

    print("CUDA context cleanup initiated. Script will now terminate.")

except Exception as e:
    print(f"Final cleanup error: {e}")


import sys

sys.exit(0)  # 정상 종료 코드 반환

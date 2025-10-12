import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM
import json

# ===================================================================
# 프로젝트 루트 경로 설정
# ===================================================================
JSONL_FILE = Path("output/Emotion_EEG/Jsonl_For_Llama3/Inference_Data.jsonl")

# LoRA 어댑터 경로
OUTPUT_DIR = Path("output/Emotion_EEG/Llama3_Result")

# ===================================================================
# 0. 설정 변수
# ===================================================================
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA 어댑터가 적용된 모델 로드
model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,  # 어댑터 저장 경로
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# 기본 모델의 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ===================================================================
# 2. 데이터 로드 (하나의 새로운 입력)
# ===================================================================

# JSONL의 첫 번째 유효 라인만 사용 (필요시 반복 처리로 확장 가능)
with JSONL_FILE.open("r", encoding="utf-8") as f:
    first = None
    for line in f:
        line = line.strip()
        if line:
            first = json.loads(line)
            break

if first is None:
    raise RuntimeError("JSONL에 유효한 레코드가 없습니다.")

orig_messages = first.get("messages", [])
if not orig_messages:
    raise RuntimeError("레코드에 'messages' 필드가 없습니다.")

# system + user만 사용해서 프롬프트 구성
messages = [m for m in orig_messages if m.get("role") in ("system", "user")]

# ===================================================================
# 3. 텍스트 생성
# ===================================================================

# 3.1 토크나이징 및 GPU 이동
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)


# 3.2 텍스트 생성
print("\n--- Generating Report ---")
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
)

# 3.3 결과 디코딩 및 추출
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

# 응답 텍스트 추출
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n"
if ASSISTANT_HEADER in generated_text:
    response = generated_text.split(ASSISTANT_HEADER)[1]
    response = response.split("<|eot_id|>")[0].strip()
    response = response.split("<|end_of_text|>")[0].strip()
else:
    response = "응답 추출 실패: 예상된 'assistant' 헤더를 찾을 수 없습니다."


print("\n**[GENERATED REPORT]**")
print("--------------------------------------------------")
print(response)
print("--------------------------------------------------")

import torch
import sys
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, AutoPeftModelForCausalLM

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
# 0. 설정 변수
# ===================================================================
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = constants.LLAMA3_ADAPTER

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ===================================================================
# 2. 테스트 데이터 준비 (하나의 새로운 입력)
# ===================================================================

# 이 형식은 훈련 데이터셋의 'user' 프롬프트와 동일해야 합니다.
NEW_EEG_DATA = """
다음 정보를 바탕으로 2~3문장 한국어 보고서 톤으로 요약하세요.
- step2.emotion_color: Happy
- step3.fill_rate: Half
- step4.fill_rate: Low
- EEG(final=step4): stress=0.15, engage=0.85, relax=0.70, excite=0.90, interest=0.95, focus=0.92
- EEG(trend = step4 - step2): d_stress=-0.05, d_engage=+0.05, d_relax=+0.60, d_excite=+0.10, d_interest=+0.15, d_focus=+0.08
요건: 2~3문장, 보고서형 어체(…로 해석됩니다/보입니다), 핵심 요소(감정·신체감각·최종 EEG·변화·복합지표)를 반드시 포함.
"""

# Llama 3.1 Instruct 모델 형식에 맞게 프롬프트 구성
messages = [
    {
        "role": "system",
        "content": "너는 VR 감정/EEG 데이터를 2~3문장으로 요약하는 한국어 보고서 작성 도우미다. 반드시 보고서형 어체를 사용하고, 과장·추측을 피하며, 입력된 지표(최종값과 변화)를 반영한다. 인지/몰입·각성/관여·조절/안정 각 그룹에서 1개씩 대표 지표를 선택해 기술하고, 전반적인 상태를 포함하라.",
    },
    {"role": "user", "content": NEW_EEG_DATA},
]

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


print(f"**[NEW EEG DATA INPUT]**\n{NEW_EEG_DATA.strip()}")
print("\n**[GENERATED REPORT]**")
print("--------------------------------------------------")
print(response)
print("--------------------------------------------------")

import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM
import json
import os


# ===================================================================
# 프로젝트 루트 경로 설정
# ===================================================================
JSONL_FILE = Path("output/Emotion_EEG/Jsonl_For_Llama3/Inference_Data.jsonl")

# LoRA 어댑터 경로
OUTPUT_DIR = Path("output/Emotion_EEG/Llama3_Result")
REPORT_OUTPUT_PATH = OUTPUT_DIR / "Generated_Report.txt"  # 생성된 보고서 저장 경로 추가


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


# ===================================================================
# 메인 실행 함수
# ===================================================================
def run_llama_inference():
    """
    미세 조정된 Llama 모델을 로드하고, JSONL 파일의 첫 번째 프롬프트를 사용하여
    보고서를 생성한 후 결과를 저장합니다.
    """

    print("Llama3Main: LLM 추론 시작...")

    # 0. 출력 폴더 확인 및 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 모델 및 토크나이저 설정
    try:
        # LoRA 어댑터가 적용된 모델 로드
        model = AutoPeftModelForCausalLM.from_pretrained(
            OUTPUT_DIR,  # 어댑터 저장 경로
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    except Exception as e:
        print(
            f"Llama3Main 오류: 모델 로드 실패. '{OUTPUT_DIR}'에 어댑터가 없거나, GPU/CUDA 환경 문제일 수 있습니다. 오류: {e}"
        )
        return False

    # 기본 모델의 토크나이저 로드 (Hugging Face 토큰 필요)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(
            f"Llama3Main 오류: 토크나이저 로드 실패. Hugging Face 토큰(Llama 접근 권한)을 확인하세요. 오류: {e}"
        )
        return False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Llama3Main: 모델 및 토크나이저 로드 성공.")

    # 2. 데이터 로드 (JSONL의 첫 번째 레코드만 사용)
    try:
        with JSONL_FILE.open("r", encoding="utf-8") as f:
            first = None
            for line in f:
                line = line.strip()
                if line:
                    first = json.loads(line)
                    break
    except FileNotFoundError:
        print(
            f"Llama3Main 오류: 입력 JSONL 파일 '{JSONL_FILE}'을 찾을 수 없습니다. (JsonToJsonlMain이 선행되어야 함)"
        )
        return False

    if first is None:
        print("Llama3Main 오류: JSONL에 유효한 레코드가 없습니다.")
        return False

    orig_messages = first.get("messages", [])
    if not orig_messages:
        print("Llama3Main 오류: 레코드에 'messages' 필드가 없습니다.")
        return False

    # system + user만 사용해서 프롬프트 구성
    messages = [m for m in orig_messages if m.get("role") in ("system", "user")]

    # 3. 텍스트 생성
    try:
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

        # 4. 결과 저장
        with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"\nLlama3Main: 생성된 보고서가 저장되었습니다: {REPORT_OUTPUT_PATH}")

        return True

    except Exception as e:
        print(f"Llama3Main 오류: 텍스트 생성 중 치명적인 오류 발생. 오류: {e}")
        return False

    finally:
        # GPU 메모리 해제
        if "model" in locals() and model.device.type == "cuda":
            del model
            torch.cuda.empty_cache()
            print("Llama3Main: GPU 메모리 정리 완료.")

        # sys.exit(0)는 모듈 실행이 아닌 파이프라인 실행 시에만 사용해야 하므로 제거했습니다.


if __name__ == "__main__":
    run_llama_inference()

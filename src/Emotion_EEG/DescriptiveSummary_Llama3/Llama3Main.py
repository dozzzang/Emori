import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM
import json
import os





JSONL_FILE = Path("output/Emotion_EEG/Jsonl_For_Llama3/Inference_Data.jsonl")


OUTPUT_DIR = Path("output/Emotion_EEG/Llama3_Result")
REPORT_OUTPUT_PATH = OUTPUT_DIR / "Generated_Report.txt"  





MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)





def run_llama_inference():
           

    print("Llama3Main: LLM 추론 시작...")

    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    try:
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            OUTPUT_DIR,  
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

    
    messages = [m for m in orig_messages if m.get("role") in ("system", "user")]

    
    try:
        
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        
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

        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        
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

        
        with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"\nLlama3Main: 생성된 보고서가 저장되었습니다: {REPORT_OUTPUT_PATH}")

        return True

    except Exception as e:
        print(f"Llama3Main 오류: 텍스트 생성 중 치명적인 오류 발생. 오류: {e}")
        return False

    finally:
        
        if "model" in locals() and model.device.type == "cuda":
            del model
            torch.cuda.empty_cache()
            print("Llama3Main: GPU 메모리 정리 완료.")

        


if __name__ == "__main__":
    run_llama_inference()

import json
from pathlib import Path
import os  


INPUT_DIR = Path("output/Emotion_EEG/Report_Json_Data")
JSON_FILE_NAME = "Report_Data.json"
SRC = INPUT_DIR / JSON_FILE_NAME


OUT = Path("output/Emotion_EEG/Jsonl_For_Llama3/Inference_Data.jsonl")



def delta(a, b):
                                    
    a = 0.0 if a is None else float(a)
    b = 0.0 if b is None else float(b)
    return a - b


def sign_fmt(x, prec=2):
                                
    return f"{x:+.{prec}f}"



def build_base_record(pid: str, participant: dict):
                                                        
    steps = participant.get("steps", {})
    s2 = steps.get("step2", {})
    s3 = steps.get("step3", {})
    s4 = steps.get("step4", {})

    base_color = s2.get("emotion_color")
    step3_fill = s3.get("fill_rate")
    step4_fill = s4.get("fill_rate")

    
    final = {
        "stress": float(s4.get("stress", s2.get("stress", 0.0))),
        "engage": float(s4.get("engage", s2.get("engage", 0.0))),
        "relax": float(s4.get("relax", s2.get("relax", 0.0))),
        "excite": float(s4.get("excite", s2.get("excite", 0.0))),
        "interest": float(s4.get("interest", s2.get("interest", 0.0))),
        "focus": float(s4.get("focus", s2.get("focus", 0.0))),
    }
    
    trend = {k: delta(final[k], float(s2.get(k, 0.0))) for k in final.keys()}

    
    user = (
        "다음 정보를 바탕으로 2~3문장 한국어 보고서 톤으로 요약하세요.\n"
        f"- step2.emotion_color: {base_color}\n"
        f"- step3.fill_rate: {step3_fill}\n"
        f"- step4.fill_rate: {step4_fill}\n"
        f"- EEG(final=step4): stress={final['stress']:.2f}, engage={final['engage']:.2f}, relax={final['relax']:.2f}, "
        f"excite={final['excite']:.2f}, interest={final['interest']:.2f}, focus={final['focus']:.2f}\n"
        f"- EEG(trend = step4 - step2): "
        f"d_stress={sign_fmt(trend['stress'])}, d_engage={sign_fmt(trend['engage'])}, d_relax={sign_fmt(trend['relax'])}, "
        f"d_excite={sign_fmt(trend['excite'])}, d_interest={sign_fmt(trend['interest'])}, d_focus={sign_fmt(trend['focus'])}\n"
        "요건: 2~3문장, 보고서형 어체(…로 해석됩니다/보입니다), 핵심 요소(감정·신체감각·최종 EEG·변화·복합지표)를 반드시 포함."
    )

    
    system = (
        "너는 VR 감정/EEG 데이터를 2~3문장으로 요약하는 한국어 보고서 작성 도우미다. "
        "반드시 보고서형 어체를 사용하고, 과장·추측을 피하며, 입력된 지표(최종값과 변화)를 반영한다. "
        "인지/몰입·각성/관여·조절/안정 각 그룹에서 1개씩 대표 지표를 선택해 기술하고, 전반적인 상태를 포함하라."
    )

    
    return {
        "user_content": user,
        "system_content": system,
        "pid": pid,
    }





def run_json_to_jsonl():
           
    print("JsonToJsonlMain: JSONL 변환 시작...")

    
    try:
        with open(SRC, "r", encoding="utf-8") as f:
            src_json = json.load(f)
    except FileNotFoundError:
        print(
            f"JsonToJsonlMain 오류: 입력 JSON 파일 '{SRC}'을 찾을 수 없습니다. (RaderChart와 KeyWord가 선행되어야 함)"
        )
        return False
    except json.JSONDecodeError as e:
        print(
            f"JsonToJsonlMain 오류: JSON 파일 디코딩 오류가 발생했습니다. 파일 내용 확인 필요. 오류: {e}"
        )
        return False

    jsonl_records = []
    total_participants = len(src_json)

    
    for pid, participant_data in src_json.items():

        
        base_record = build_base_record(pid, participant_data)

        
        record = {
            "messages": [
                {"role": "system", "content": base_record["system_content"]},
                {"role": "user", "content": base_record["user_content"]},
            ],
            "meta": {
                "participant_id": pid,
                "policy": "input_for_inference",
            },
        }
        jsonl_records.append(record)

    
    try:
        
        os.makedirs(OUT.parent, exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            for r in jsonl_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"JsonToJsonlMain: 총 {total_participants}명의 데이터를 처리했습니다.")
        print(
            f"JsonToJsonlMain: {len(jsonl_records)}개의 입력 레코드가 '{OUT}'으로 생성되었습니다."
        )
        return True
    except Exception as e:
        print(f"JsonToJsonlMain 오류: JSONL 파일 저장 중 오류 발생: {e}")
        return False


if __name__ == "__main__":
    run_json_to_jsonl()

import json
import pandas as pd
import os, sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 이 경로는 실행 스크립트의 위치에 따라 조정이 필요할 수 있습니다.
project_root = os.path.join(current_script_dir, "..")

if project_root not in sys.path:
    sys.path.append(project_root)

import constants

# ===== 파일 경로 정의 (constants.py 참조) =====
# 입력 파일 경로: Report_Data.json (일반 JSON)
SRC = constants.MAIN_JSON_FILE
# 출력 파일 경로: Train_Data.jsonl (인퍼런스 입력용 JSONL로 사용)
OUT = constants.TRAIN_JSONL_FILE

# 참고: constants.ASSISTANT_LABELS 경로 및 관련 로직은 제거되었습니다.


# ===== 보조 함수 (변경 없음) =====
def delta(a, b):
    """변화량 Δ = a - b (None 안전 처리)"""
    a = 0.0 if a is None else float(a)
    b = 0.0 if b is None else float(b)
    return a - b


def sign_fmt(x, prec=2):
    """부호 포함 포맷 +0.12 / -0.07"""
    return f"{x:+.{prec}f}"


# ===== JSON → JSONL 생성 함수 (LLM 입력 프롬프트 생성) =====
def build_base_record(pid: str, participant: dict):
    """JSON 데이터를 읽어 user와 system 필드만 포함된 JSONL 뼈대를 반환"""
    steps = participant.get("steps", {})
    # step2 데이터가 없을 경우를 대비하여 기본값 {} 사용
    s2 = steps.get("step2", {})
    s3 = steps.get("step3", {})
    s4 = steps.get("step4", {})

    base_color = s2.get("emotion_color")
    step3_fill = s3.get("fill_rate")
    step4_fill = s4.get("fill_rate")

    # 최종(step4) 값 (step4에 값이 없으면 step2 값 사용, 그래도 없으면 0.0)
    final = {
        "stress": float(s4.get("stress", s2.get("stress", 0.0))),
        "engage": float(s4.get("engage", s2.get("engage", 0.0))),
        "relax": float(s4.get("relax", s2.get("relax", 0.0))),
        "excite": float(s4.get("excite", s2.get("excite", 0.0))),
        "interest": float(s4.get("interest", s2.get("interest", 0.0))),
        "focus": float(s4.get("focus", s2.get("focus", 0.0))),
    }
    # 트렌드(step4 - step2)
    trend = {k: delta(final[k], float(s2.get(k, 0.0))) for k in final.keys()}

    # ===== 입력(user) - LLM이 요약할 데이터 =====
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

    # ===== 시스템(system) - LLM의 역할 정의 =====
    system = (
        "너는 VR 감정/EEG 데이터를 2~3문장으로 요약하는 한국어 보고서 작성 도우미다. "
        "반드시 보고서형 어체를 사용하고, 과장·추측을 피하며, 입력된 지표(최종값과 변화)를 반영한다. "
        "인지/몰입·각성/관여·조절/안정 각 그룹에서 1개씩 대표 지표를 선택해 기술하고, 전반적인 상태를 포함하라."
    )

    # 이 뼈대 딕셔너리를 반환
    return {
        "user_content": user,
        "system_content": system,
        "pid": pid,
    }


def main():
    # 1. 입력 JSON 파일 로드
    try:
        with open(SRC, "r", encoding="utf-8") as f:
            src_json = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: 입력 JSON 파일 '{SRC}'을 찾을 수 없습니다. 경로를 확인해 주세요."
        )
        return
    except json.JSONDecodeError as e:
        print(
            f"Error: JSON 파일 디코딩 오류가 발생했습니다. 파일 내용 확인 필요. 오류: {e}"
        )
        return

    jsonl_records = []
    total_participants = len(src_json)

    # 2. 통합된 JSON 파일의 모든 참가자 데이터를 순회 (pid: participant_data 형식)
    for pid, participant_data in src_json.items():

        # 3. LLM에게 전달할 system 및 user 프롬프트 생성
        base_record = build_base_record(pid, participant_data)

        # 4. JSONL 인퍼런스 입력 레코드 생성 (assistant 필드 없음)
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

    # 5. JSONL 파일로 출력
    with open(OUT, "w", encoding="utf-8") as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"총 {total_participants}명의 참가자 데이터를 처리했습니다.")
    print(
        f"총 {len(jsonl_records)}개의 인퍼런스 입력 레코드가 '{OUT}'으로 생성되었습니다."
    )


if __name__ == "__main__":
    main()

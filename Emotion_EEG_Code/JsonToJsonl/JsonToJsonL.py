import json
import pandas as pd
import os, sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 이 경로는 실행 스크립트의 위치에 따라 조정이 필요할 수 있습니다.
project_root = os.path.join(current_script_dir, "..")

if project_root not in sys.path:
    sys.path.append(project_root)

import constants

SRC = constants.OUTPUT_JSON_FILE
LABEL_CSV = constants.ASSISTANT_LABELS
OUT = constants.TRAIN_JSONL_FILE


# ===== 보조 함수 =====
def delta(a, b):
    """변화량 Δ = a - b (None 안전 처리)"""
    a = 0.0 if a is None else float(a)
    b = 0.0 if b is None else float(b)
    return a - b


def sign_fmt(x, prec=2):
    """부호 포함 포맷 +0.12 / -0.07"""
    return f"{x:+.{prec}f}"


# ===== JSON → JSONL 생성 함수 =====
def build_base_record(pid: str, participant: dict):
    """JSON 데이터를 읽어 user와 system 필드만 포함된 JSONL 뼈대를 반환"""
    steps = participant.get("steps", {})
    s2, s3, s4 = steps.get("step2", {}), steps.get("step3", {}), steps.get("step4", {})

    base_color = s2.get("emotion_color")
    step3_fill = s3.get("fill_rate")
    step4_fill = s4.get("fill_rate")

    # 최종(step4) 값
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

    # ===== 입력(user) =====
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

    # ===== 시스템(system) =====
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


def load_manual_labels(csv_file_path: str) -> dict:
    """CSV 파일에서 수동 작성된 정답지(Assistant)를 로드하여 딕셔너리로 그룹화"""
    try:
        # CSV 파일 로드
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: 정답 파일 '{csv_file_path}'을 찾을 수 없습니다.")
        return {}
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}

    # 참가자 ID(pid)별로 정답(assistant_content)을 리스트로 그룹화
    label_map = {}
    for pid, group in df.groupby("participant_id"):
        # 'summary_ko' 열의 값만 리스트로 변환하여 저장
        label_map[pid] = group["summary_ko"].tolist()

    return label_map


def main():
    # 1. 수동 작성된 정답 목록 로드 (CSV 파일 사용)
    manual_labels_map = load_manual_labels(LABEL_CSV)
    if not manual_labels_map:
        print("정답 데이터가 없으므로 JSONL 생성을 중단합니다.")
        return

    # 2. 입력 JSON 파일 로드
    with open(SRC, "r", encoding="utf-8") as f:
        src_json = json.load(f)

    jsonl_records = []
    missing_labels_count = 0

    # 3. 통합된 JSON 파일의 모든 참가자 데이터를 순회
    for pid, participant_data in src_json.items():
        # 해당 참가자에 대한 수동 정답 목록을 로드
        assistant_answers = manual_labels_map.get(pid)

        if not assistant_answers or len(assistant_answers) < 4:
            # 4개 미만의 정답이 있을 경우 경고 및 건너뛰기
            missing_labels_count += 1
            continue

        # 4. 입력(user) 및 시스템(system)의 뼈대를 먼저 생성
        base_record = build_base_record(pid, participant_data)

        # 5. N개의 정답 목록을 반복하면서 레코드를 생성하고 추가
        for idx, answer in enumerate(assistant_answers):
            record = {
                "messages": [
                    {"role": "system", "content": base_record["system_content"]},
                    {"role": "user", "content": base_record["user_content"]},
                    {
                        "role": "assistant",
                        "content": answer,
                    },  # CSV에서 로드된 정답 삽입
                ],
                "meta": {
                    "participant_id": pid,
                    "policy": f"final+trend_ver{idx+1}",
                },
            }
            jsonl_records.append(record)

    # 6. JSONL 파일로 출력
    with open(OUT, "w", encoding="utf-8") as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"총 {len(src_json)}명의 참가자 중 {missing_labels_count}명이 레이블링 부족으로 제외되었습니다."
    )
    print(f"총 {len(jsonl_records)}개의 학습 레코드가 '{OUT}'으로 생성되었습니다.")


if __name__ == "__main__":
    main()

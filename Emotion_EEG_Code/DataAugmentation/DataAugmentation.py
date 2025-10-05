import json
import re
import random
import os
import sys
from typing import Dict, Any, Tuple

# 현재 스크립트 파일의 절대 경로를 가져옵니다.
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 'Emotion_EEG_Code' 폴더를 포함하고 있는 프로젝트의 '루트' 폴더를 계산
# 스크립트 위치가 루트보다 아래인 경우 스크립트를 실행하는 위치의 상위 경로(부모 디렉터리)를 프로젝트 루트로 간주
project_root = os.path.join(current_script_dir, "..")

# 파이썬 경로에 프로젝트 루트를 추가합니다.
if project_root not in sys.path:
    sys.path.append(project_root)

import constants

OUTPUT_JSON_FILE = constants.OUTPUT_JSON_FILE
BASE_INPUT_FILE = constants.BASE_INPUT_FILE


### 서술형 요약 기능 구현을 위한 ai 학습용 데이터 마련을 위한 증강 코드 ###


# ===== 상수 설정 및 파일 경로 정의 =====
# index 9까지의 데이터는 원본 데이터의 흐름 유지
EMOTION_CHOICES = ["Happy", "Fear", "Sad", "Surprise", "Angry", "Disgust"]
FILL_RATE_CHOICES_STEP2 = ["Full", "High", "Half", "Low"]
FILL_RATE_CHOICES_STEP3 = ["Full", "High", "Half", "Low", "Minimal"]
PM_AUGMENT_RANGE = 0.2  # PM 값 변동 범위: +/- 20% (index < 9 에서 사용)
PM_CORRELATION_NOISE = 0.2  # 상관관계 기반 값의 변동 폭 (index >= 9 에서 사용)
PM_NEAR_ZERO_THRESHOLD = 0.05  # 이 값보다 작으면 0 근처에서 변동 (index < 9 에서 사용)
PM_NEAR_ZERO_MAX = 0.1  # (index < 9 에서 사용)


# ===== 값 추출 및 보조 함수 =====
def extract_line_value(label: str, text: str, null_value=None) -> Any:
    """TXT 파일에서 단일 라벨 값 추출"""
    lab = label.strip()
    for line in text.splitlines():
        m = re.match(rf"^\s*{re.escape(lab)}\s*:\s*(.*)\s*$", line)
        if m:
            val = m.group(1).strip()
            return val if val else null_value
    return null_value


def extract_emotion(pattern, text, flags=0):
    """정규식을 사용하여 감정/채우기 비율 값 추출"""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def make_metrics(stress, engage, relax, excite, interest, focus):
    """PM 메트릭스 딕셔너리 생성 (PM_ 제거된 키 사용)"""
    return {
        "stress": stress,
        "engage": engage,
        "relax": relax,
        "excite": excite,
        "interest": interest,
        "focus": focus,
    }


def extract_base_data(raw_content: str) -> Dict[str, Any]:
    data = {}

    # UserInfo/OwnerInfo 추출
    data["NAME"] = extract_line_value("NAME", raw_content)
    data["AGE"] = extract_line_value("AGE", raw_content)
    data["GENDER"] = extract_line_value("GENDER", raw_content)
    data["REPORT_DAY"] = extract_line_value("REPORT_DAY", raw_content)

    # GameData 추출
    data["STEP1_EMOTION_COLOR"] = extract_emotion(
        r"STEP1_EMOTION_COLOR\s*:\s*(.*)", raw_content
    )
    data["STEP2_FILL_RATE"] = extract_emotion(
        r"STEP2_FILL_RATE\s*:\s*(.*)", raw_content
    )
    data["STEP3_FILL_RATE"] = extract_emotion(
        r"STEP3_FILL_RATE\s*:\s*(.*)", raw_content
    )

    # Step별 PM 수치 추출
    pm_by_state = {}
    target_states = ["Step1", "Step2", "Step3"]
    pm_keys_full = [
        "PM_Stress",
        "PM_Engage",
        "PM_Relax",
        "PM_Excite",
        "PM_Interest",
        "PM_Focus",
    ]

    # TXT 파일의 STEP 블록을 모두 찾음
    step_blocks = re.findall(
        r"-{5,}STEP\.([A-Za-z0-9_]+)-{5,}\s*(.*?)(?=(?:-{5,}STEP\.[A-Za-z0-9_]+-{5,})|$)",
        raw_content,
        flags=re.DOTALL,
    )

    for state, body in step_blocks:
        if state not in target_states:
            continue

        vals = {}
        # 해당 Step에서 첫 번째 PM 값 블록만 사용
        for key in pm_keys_full:
            m = re.search(rf"{key}\s*:\s*([0-9]*\.?[0-9]+)", body)
            if m:
                try:
                    vals[key] = float(m.group(1))
                except ValueError:
                    vals[key] = 0.0
            else:
                vals[key] = 0.0

        if vals and not pm_by_state.get(state):
            # PM_이 없는 최종 키로 저장 (예: 'stress', 'engage'...)
            pm_by_state[state] = make_metrics(
                vals.get("PM_Stress", 0.0),
                vals.get("PM_Engage", 0.0),
                vals.get("PM_Relax", 0.0),
                vals.get("PM_Excite", 0.0),
                vals.get("PM_Interest", 0.0),
                vals.get("PM_Focus", 0.0),
            )

    data["PM_Step1"] = pm_by_state.get("Step1", make_metrics(0, 0, 0, 0, 0, 0))
    data["PM_Step2"] = pm_by_state.get("Step2", make_metrics(0, 0, 0, 0, 0, 0))
    data["PM_Step3"] = pm_by_state.get("Step3", make_metrics(0, 0, 0, 0, 0, 0))

    return data


# --- 제약 조건을 적용한 노이즈 생성 ---
def get_constrained_value(base_val, noise_factor=PM_CORRELATION_NOISE):
    """기반 값 주변에 노이즈를 적용하고 0.001~1.0 범위를 유지"""
    min_val = max(0.001, base_val * (1 - noise_factor))
    max_val = min(1.0, base_val * (1 + noise_factor))
    return round(random.uniform(min_val, max_val), 7)


# ===== 데이터 증강 =====
def augment_data(base_data: Dict[str, Any], index: int) -> Tuple[str, Dict[str, Any]]:
    """기본 데이터를 기반으로 값을 무작위로 변경하여 새로운 참가자 데이터를 생성"""

    new_data = json.loads(json.dumps(base_data))

    # 식별 정보 변경
    base_file_id = os.path.basename(BASE_INPUT_FILE).replace(".txt", "")
    participant_id = f"participant_{base_file_id}_AUG{index+1:03d}"

    # GameData 무작위 변경
    new_data["STEP1_EMOTION_COLOR"] = random.choice(EMOTION_CHOICES)
    new_data["STEP2_FILL_RATE"] = random.choice(FILL_RATE_CHOICES_STEP2)
    new_data["STEP3_FILL_RATE"] = random.choice(FILL_RATE_CHOICES_STEP3)

    # --- PM 값 무작위 조절 ---
    for step_key in ["PM_Step1", "PM_Step2", "PM_Step3"]:

        if index >= 9:
            # 1. 정서 기반 (Relax/Stress 관계 정의)
            # Relax Base는 0.1~0.9 범위에서 생성하여 극단적인 0/1 상태를 약간 피함
            relax_base = random.uniform(0.1, 0.9)

            # Stress Base는 Relax의 반대 (1 - Relax)
            stress_base = 1.0 - relax_base

            # 2. 인지 기반 (Engage, Interest, Focus 관계 정의)
            # Cognitive Base는 0.001~1.0 사이에서 생성
            cognitive_base = random.uniform(0.001, 1.0)

        pm_block = new_data[step_key]

        for key in pm_block.keys():

            if index < 9:
                # AUG001 ~ AUG009 : 원본의 로직 유지
                original_value = pm_block.get(key, 0.5)

                # 노이즈 범위 설정
                noise_min = max(0.0, original_value * (1 - PM_AUGMENT_RANGE))
                noise_max = min(1.0, original_value * (1 + PM_AUGMENT_RANGE))

                # 0에 가까운 값 처리
                if original_value < PM_NEAR_ZERO_THRESHOLD:
                    new_value = round(random.uniform(0.001, PM_NEAR_ZERO_MAX), 7)
                else:
                    new_value = round(random.uniform(noise_min, noise_max), 7)

            else:
                # AUG010 이상 (index 9 ~ 끝): 제약 조건 기반 랜덤 값 적용
                if key == "relax":
                    new_value = get_constrained_value(relax_base)
                elif key == "stress":
                    # Relax 기반 값 주변에서 변동
                    new_value = get_constrained_value(stress_base)
                elif key in ["engage", "interest", "focus"]:
                    # Cognitive 기반 값 주변에서 변동
                    new_value = get_constrained_value(cognitive_base)
                elif key == "excite":
                    # Excite는 독립적인 랜덤 값을 유지 (각성 다양성 부여)
                    new_value = round(random.uniform(0.001, 1.0), 7)
                else:
                    # 안전을 위한 기본값
                    new_value = round(random.uniform(0.001, 1.0), 7)

            pm_block[key] = new_value

    # 최종 JSON 구조에 맞춰 데이터 구성
    participant_data = {
        "basic_info": {
            "age": new_data.get("AGE"),
            "gender": new_data.get("GENDER"),
            "date": new_data.get("REPORT_DAY"),
        },
        "steps": {
            "step2": {
                "emotion_color": new_data["STEP1_EMOTION_COLOR"],
                **new_data["PM_Step1"],
            },
            "step3": {
                "fill_rate": new_data["STEP2_FILL_RATE"],
                **new_data["PM_Step2"],
            },
            "step4": {
                "fill_rate": new_data["STEP3_FILL_RATE"],
                **new_data["PM_Step3"],
            },
        },
    }

    return participant_id, participant_data



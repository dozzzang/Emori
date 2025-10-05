import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm

# ====== Path ======
# JSON 파일을 읽어올 상위 디렉토리 (예: Path("C:/data/input_files"))
DATA_DIR = Path(".")
# 차트 이미지를 저장할 상위 디렉토리 (예: Path("C:/data/output_charts"))
OUTPUT_DIR = Path(".")

# 데이터 입력 파일 경로 (DATA_DIR에 위치)
JSON_PATH = DATA_DIR / "Report_Data.json"
# 차트 출력 파일 경로 (OUTPUT_DIR에 위치)
BAR_OUT_PATH = OUTPUT_DIR / "bar_chart.png"
RADAR_OUT_PATH = OUTPUT_DIR / "radar_chart.png"

# 기타 Configuration
STEPS_TO_PLOT = ["step2", "step3", "step4"]
BAR_COLORS = ["#6BAED6", "#74C476", "#FD8D3C"]
RADAR_COLOR = "darkorange"
BAR_WIDTH = 0.25
# ====== JSON 로드 ======
try:
    print(f"JSON 파일 로드 시도: {JSON_PATH.resolve()}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    # 경로 설정 정보 추가 출력
    print(f"오류: {JSON_PATH.resolve()} 파일을 찾을 수 없습니다.")
    print(f"경로: '{DATA_DIR}'와 파일명: 'Report_Data.json'을 확인하세요.")
    exit()
except json.JSONDecodeError:
    print(f"오류: {JSON_PATH} 파일의 JSON 형식이 올바르지 않습니다.")
    exit()
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")
    exit()

try:
    participant_key = next(iter(data.keys()))
    steps_data = data[participant_key]["steps"]
except (StopIteration, KeyError):
    print("오류: JSON 파일에 참가자 데이터 또는 'steps' 데이터가 없습니다.")
    exit()
# ====== 5가지 지표 계산 ======
def clamp01(x):
    """값을 0.0과 1.0 사이로 제한합니다."""
    # float(x)를 사용하여 안전하게 숫자로 변환
    try:
        x_float = float(x)
    except (ValueError, TypeError):
        x_float = 0.0  # 유효하지 않은 값은 0으로 처리

    return max(0.0, min(1.0, x_float))


def calculate_indices(m):
    stress = clamp01(m.get("stress", 0.0))
    engage = clamp01(m.get("engage", 0.0))
    relax = clamp01(m.get("relax", 0.0))
    excite = clamp01(m.get("excite", 0.0))
    interest = clamp01(m.get("interest", 0.0))
    focus = clamp01(m.get("focus", 0.0))

    """
    인지 부하: 스트레스와 이완 부족이 결합된 정신적 부담의 정도를 측정하며, 점수가 낮을수록 심리적으로 안정된 상태
    정서적 긍정성: 긍정적 감정이 부정적 스트레스보다 얼마나 우위에 있는지 나타내며, 점수가 높을수록 정서적 만족도가 높음
    주도적 집중: 단순히 집중하는 것을 넘어, 능동적인 참여가 반영된 집중 상태를 나타냄
    이완-활력 균형: 사용자의 심리적 안정 상태를 나타내며, 이완과 각성이 조화로울수록 정서적으로 안정된 상태임을 의미
    종합 몰입도: 체험에 대한 몰입, 집중, 흥미 세 가지 핵심 요소의 전반적인 평균 참여도를 나타냄
    """
    indices = {
        "인지 부하": (stress + (1 - relax)) / 2,
        "정서적 긍정성": (interest + excite - stress) / 3,
        "주도적 집중": 0.6 * engage + 0.4 * focus,
        "이완-활력\n균형": 1 - abs(relax - excite),
        "종합 몰입도": (engage + focus + interest) / 3,
    }

    # 최종 결과도 0.0 ~ 1.0 사이로 클램프
    for k in indices:
        indices[k] = float(np.clip(indices[k], 0.0, 1.0))

    return indices


# 모든 단계의 지표 계산 및 저장
all_step_indices = {}
for step_name in STEPS_TO_PLOT:
    if step_name in steps_data:
        m = steps_data[step_name]
        all_step_indices[step_name] = calculate_indices(m)
    else:
        print(f"경고: {step_name} 데이터가 JSON 파일에 없습니다. 건너뜁니다.")



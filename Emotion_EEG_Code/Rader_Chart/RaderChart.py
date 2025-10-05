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

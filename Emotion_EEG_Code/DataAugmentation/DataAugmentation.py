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



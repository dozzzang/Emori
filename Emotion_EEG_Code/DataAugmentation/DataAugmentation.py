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
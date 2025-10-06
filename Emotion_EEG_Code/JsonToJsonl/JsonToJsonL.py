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



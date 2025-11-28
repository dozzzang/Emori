# EEG_Table_Visualizer.py

import os
import re
import glob
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager, rcParams


# 폰트 설정
korean_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumGothicCoding"]

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for f in korean_fonts:
    if f in available_fonts:
        rcParams["font.family"] = f
        break

# 마이너스 기호 깨짐 방지
rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------
# 1. 공통 설정
# ------------------------------------------------------------------

DIR_PATH = "data/Emotion_EEG/VR_Result_Data"
OUTPUT_DIR = "output/Emotion_EEG/EEG_Tables"

# 기준뇌파 열 제거: Step2, Step3, Step4만 사용
STEP_CONFIG = [
    ("step2", "STEP2", ["Step2"]),
    ("step3", "STEP3", ["Step3"]),
    ("step4", "STEP4", ["Step4"]),
]

# 뇌파 지표
METRIC_CONFIG = [
    ("stress", "스트레스"),
    ("engage", "참여"),
    ("relax", "이완"),
    ("excite", "자극"),
    ("interest", "관심"),
    ("focus", "집중"),
]

# 스트레스: 핑크, 참여: 민트, 이완: 연두, 자극: 노랑, 관심: 주황, 집중: 하늘색
METRIC_BASE_COLORS = {
    "stress": "#FF69B4",
    "engage": "#48C9B0",
    "relax": "#82E0AA",
    "excite": "#F4D03F",
    "interest": "#F39C12",
    "focus": "#5DADE2",
}


# ------------------------------------------------------------------
# EEG 추출 로직
# ------------------------------------------------------------------
def extract_line_value(label: str, text: str, null_value="") -> str:
    """한 줄에 'LABEL: 값' 구조로 있는 데이터를 추출."""
    lab = label.strip()
    for line in text.splitlines():
        m = re.match(rf"^\s*{re.escape(lab)}\b\s*:\s*(.*)\s*$", line)
        if m:
            val = m.group(1).strip()
            return val if val else null_value
    return null_value


def extract_pm_by_state(raw: str) -> Dict[str, List[Dict[str, float]]]:
    """
    TXT 전체 문자열에서 STEP.<State> 블록을 찾아
    PM_Stress ~ PM_Focus 값을 state 별로 모은다.
    (TxtsToJson.py와 동일한 방식)
    """
    step_blocks = re.findall(
        r"-{5,}STEP\.([A-Za-z0-9_]+)-{5,}\s*(.*?)(?=(?:-{5,}STEP\.[A-Za-z0-9_]+-{5,})|$)",
        raw,
        flags=re.DOTALL,
    )

    pm_by_state: Dict[str, List[Dict[str, float]]] = {}

    for state, body in step_blocks:
        vals = {}
        for key in [
            "PM_Stress",
            "PM_Engage",
            "PM_Relax",
            "PM_Excite",
            "PM_Interest",
            "PM_Focus",
        ]:
            m = re.search(rf"{key}\s*:\s*([0-9]*\.?[0-9]+)", body)
            if m:
                vals[key] = float(m.group(1))

        if vals:
            pm_by_state.setdefault(state, []).append(vals)

    return pm_by_state


def make_metrics(stress, engage, relax, excite, interest, focus) -> Dict[str, float]:
    """단일 state의 뇌파 dict를 통일된 키로 변환."""
    return {
        "stress": stress,
        "engage": engage,
        "relax": relax,
        "excite": excite,
        "interest": interest,
        "focus": focus,
    }


def metrics_of_state(
    state_name: str,
    pm_by_state: Dict[str, List[Dict[str, float]]],
    fallback: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    """
    특정 state_name에 대한 뇌파 수치를 반환.
    여러 값이 있을 경우 첫 번째 값만 사용 (TxtsToJson.py와 동일).
    """
    arr = pm_by_state.get(state_name, [])
    if not arr:
        return fallback

    x = arr[0]
    s = x.get("PM_Stress", 0.0)
    eg = x.get("PM_Engage", 0.0)
    r = x.get("PM_Relax", 0.0)
    ex = x.get("PM_Excite", 0.0)
    i = x.get("PM_Interest", 0.0)
    f = x.get("PM_Focus", 0.0)

    return make_metrics(s, eg, r, ex, i, f)


def metrics_from_candidates(
    candidates: List[str], pm_by_state: Dict[str, List[Dict[str, float]]]
) -> Optional[Dict[str, float]]:
    """
    state 이름 후보가 여러 개일 때,
    가장 먼저 발견되는 state의 metrics를 반환.
    """
    for name in candidates:
        m = metrics_of_state(name, pm_by_state, fallback=None)
        if m is not None:
            return m
    return None


# ------------------------------------------------------------------
# 3. 원형 그라데이션 그리기
# ------------------------------------------------------------------
def quintile_level(v: float) -> float:
    """
    0~1 값 v를 5개 구간으로 나누어
    0.2, 0.4, 0.6, 0.8, 1.0 중 하나의 단계로 반환.
    """
    if v <= 0.2:
        return 0.2
    elif v <= 0.4:
        return 0.4
    elif v <= 0.6:
        return 0.6
    elif v <= 0.8:
        return 0.8
    else:
        return 1.0


def plot_eeg_table(
    participant_id: str, stage_to_metrics: Dict[str, Dict[str, float]], save_path: str
):
    """
    stage_to_metrics:
        {
            "step1": {"stress": ..., "engage": ..., ...},
            "step2": {...},
            "step3": {...},
        }
    를 이용하여 표 + 그라데이션 색상의 원을 그림.
    """

    # 열(지표), 행(단계) 순서
    stage_keys = [s[0] for s in STEP_CONFIG]  # y축
    stage_labels = [s[1] for s in STEP_CONFIG]

    metric_keys = [m[0] for m in METRIC_CONFIG]  # x축
    metric_labels = [m[1] for m in METRIC_CONFIG]

    values = np.zeros((len(metric_keys), len(stage_keys)))
    mask = np.zeros_like(values, dtype=bool)

    for j, sk in enumerate(stage_keys):
        metrics = stage_to_metrics.get(sk)
        if metrics is None:
            mask[:, j] = True
            continue
        for i, mk in enumerate(metric_keys):
            val = metrics.get(mk)
            if val is None:
                mask[i, j] = True
            else:
                values[i, j] = val

    # 시각화를 위한 정규화: 각 지표별로 min~max 기준 [0,1]로 스케일
    norm_values = values.copy()
    for i in range(len(metric_keys)):
        row = values[i, :]
        valid = ~mask[i, :]
        if not valid.any():
            continue
        vmin = row[valid].min()
        vmax = row[valid].max()
        if vmax - vmin < 1e-8:
            norm_values[i, valid] = 0.5  # 모두 동일한 값인 경우 중간값
        else:
            norm_values[i, valid] = (row[valid] - vmin) / (vmax - vmin)

    # 그림 생성
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(-0.5, len(metric_keys) - 0.5)
    ax.set_ylim(-0.5, len(stage_keys) - 0.5)
    ax.invert_yaxis()

    # 축 눈금/레이블
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticks(range(len(stage_keys)))
    ax.set_yticklabels(stage_labels, fontsize=11)

    ax.tick_params(axis="both", length=0)

    # 격자선, 테두리 설정
    for x in range(len(metric_keys) + 1):
        ax.axvline(x - 0.5, color="black", linewidth=1.0)
    for y in range(len(stage_keys) + 1):
        ax.axhline(y - 0.5, color="black", linewidth=1.0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # 각 셀에 원 그리기
    for i, mk in enumerate(metric_keys):
        base_rgb = mcolors.to_rgb(METRIC_BASE_COLORS[mk])
        for j, sk in enumerate(stage_keys):
            if mask[i, j]:
                continue

            v = norm_values[i, j]  # 0 ~ 1 (해당 지표 내에서 상대값)

            level = quintile_level(v)

            # 흰색(1,1,1)과 base_rgb 사이를 level 비율로 보간
            color = tuple((1.0 - level) * 1.0 + level * c for c in base_rgb)

            # x = 지표 index(i), y = 단계 index(j)
            ax.scatter(
                i,
                j,
                s=600,
                color=color,
                edgecolor="none",
            )

    ax.set_title(f"EEG 변화 ({participant_id})", fontsize=12)
    ax.set_aspect("equal")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def debug_plot_quintile_example(
    save_path: str = os.path.join(OUTPUT_DIR, "quintile_example.png")
):
    """
    5분위 색상이 어떻게 표현되는지 보기 위한 단순 예시 PNG를 생성.
    예시로 스트레스(핑크 계열) 하나만 사용하고,
    v = 0.1, 0.3, 0.5, 0.7, 0.9 에 대해 색을 찍어보는 예시.
    """
    metric_key = "stress"

    base_rgb = mcolors.to_rgb(METRIC_BASE_COLORS[metric_key])
    v_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    labels = ["0~20%", "21~40%", "41~60%", "61~80%", "81~100%"]

    n = len(v_values)
    fig, ax = plt.subplots(figsize=(6, 2))

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, 0.5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([])
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, v in enumerate(v_values):
        level = quintile_level(v)
        color = tuple((1.0 - level) * 1.0 + level * c for c in base_rgb)

        ax.scatter(i, 0, s=800, color=color, edgecolor="none")
        ax.text(
            i,
            0.25,
            f"v={v:.1f}\nL={int(level*100)}%",
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_title("5분위 색상 예시 (지표: 스트레스/핑크 계열)", fontsize=11)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------
# 4. 메인 실행: txt → EEG 테이블 이미지
# ------------------------------------------------------------------
def main():
    file_pattern = os.path.join(DIR_PATH, "**", "RECORD*.txt")
    file_list = glob.glob(file_pattern, recursive=True)

    if not file_list:
        print(f"경로 '{DIR_PATH}'에서 RECORD*.txt 파일을 찾지 못했습니다.")
        return

    print(f"총 {len(file_list)}개의 RECORD*.txt 파일에 대해 EEG 테이블을 생성합니다.")

    for path in sorted(file_list):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        base_name = os.path.splitext(os.path.basename(path))[0]
        participant_id = base_name

        pm_by_state = extract_pm_by_state(raw)

        # 단계별 EEG metric 추출
        stage_to_metrics: Dict[str, Dict[str, float]] = {}
        for key, _, candidates in STEP_CONFIG:
            metrics = metrics_from_candidates(candidates, pm_by_state)
            if metrics is not None:
                stage_to_metrics[key] = metrics

        if not stage_to_metrics:
            print(f"[경고] {participant_id}에서 사용할 뇌파 데이터가 없습니다.")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{participant_id}_EEG_Table.png")
        plot_eeg_table(participant_id, stage_to_metrics, out_path)
        print(f"  → {out_path} 생성 완료")


if __name__ == "__main__":
    # debug_plot_quintile_example()
    main()

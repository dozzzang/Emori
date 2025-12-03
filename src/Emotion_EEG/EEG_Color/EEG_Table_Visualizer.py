

import os
import re
import glob
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager, rcParams



korean_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumGothicCoding"]

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for f in korean_fonts:
    if f in available_fonts:
        rcParams["font.family"] = f
        break


rcParams["axes.unicode_minus"] = False





DIR_PATH = "data/Emotion_EEG/VR_Result_Data"
OUTPUT_DIR = "output/Emotion_EEG/EEG_Tables"


STEP_CONFIG = [
    ("step2", "STEP2", ["Step1"]),
    ("step3", "STEP3", ["Step2"]),
    ("step4", "STEP4", ["Step3"]),
]


METRIC_CONFIG = [
    ("stress", "스트레스"),
    ("engage", "참여"),
    ("relax", "이완"),
    ("excite", "자극"),
    ("interest", "관심"),
    ("focus", "집중"),
]


METRIC_BASE_COLORS = {
    "stress": "#FF69B4",
    "engage": "#48C9B0",
    "relax": "#82E0AA",
    "excite": "#F4D03F",
    "interest": "#F39C12",
    "focus": "#5DADE2",
}





def extract_line_value(label: str, text: str, null_value="") -> str:
                                         
    lab = label.strip()
    for line in text.splitlines():
        m = re.match(rf"^\s*{re.escape(lab)}\b\s*:\s*(.*)\s*$", line)
        if m:
            val = m.group(1).strip()
            return val if val else null_value
    return null_value


def extract_pm_by_state(raw: str) -> Dict[str, List[Dict[str, float]]]:
           
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
           
    for name in candidates:
        m = metrics_of_state(name, pm_by_state, fallback=None)
        if m is not None:
            return m
    return None





def quintile_level(v: float) -> float:
           
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
           

    
    stage_keys = [s[0] for s in STEP_CONFIG]  
    stage_labels = [s[1] for s in STEP_CONFIG]

    metric_keys = [m[0] for m in METRIC_CONFIG]  
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

    
    norm_values = values.copy()
    for i in range(len(metric_keys)):
        row = values[i, :]
        valid = ~mask[i, :]
        if not valid.any():
            continue
        vmin = row[valid].min()
        vmax = row[valid].max()
        if vmax - vmin < 1e-8:
            norm_values[i, valid] = 0.5  
        else:
            norm_values[i, valid] = (row[valid] - vmin) / (vmax - vmin)

    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(-0.5, len(metric_keys) - 0.5)
    ax.set_ylim(-0.5, len(stage_keys) - 0.5)
    ax.invert_yaxis()

    
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticks(range(len(stage_keys)))
    ax.set_yticklabels(stage_labels, fontsize=11)

    ax.tick_params(axis="both", length=0)

    
    for x in range(len(metric_keys) + 1):
        ax.axvline(x - 0.5, color="black", linewidth=1.0)
    for y in range(len(stage_keys) + 1):
        ax.axhline(y - 0.5, color="black", linewidth=1.0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    
    for i, mk in enumerate(metric_keys):
        base_rgb = mcolors.to_rgb(METRIC_BASE_COLORS[mk])
        for j, sk in enumerate(stage_keys):
            if mask[i, j]:
                continue

            v = norm_values[i, j]  

            level = quintile_level(v)

            
            color = tuple((1.0 - level) * 1.0 + level * c for c in base_rgb)

            
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
    
    main()

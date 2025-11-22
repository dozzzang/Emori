import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm

# ====== Path & Configuration ======
INPUT_DIR = Path("output/Emotion_EEG/Report_Json_Data")
OUTPUT_DIR = Path("output/Emotion_EEG/Chart_Result")

JSON_PATH = INPUT_DIR / "Report_Data.json"
BAR_OUT_PATH = OUTPUT_DIR / "bar_chart.png"
RADAR_OUT_PATH = OUTPUT_DIR / "radar_chart.png"

# 기타 Configuration
STEPS_TO_PLOT = ["step2", "step3", "step4"]
BAR_COLORS = ["#6BAED6", "#74C476", "#FD8D3C"]
RADAR_COLOR = "darkorange"
BAR_WIDTH = 0.25


# ====== Matplotlib 폰트 설정 (한글 깨짐 방지) ======
def set_korean_font():
    """시스템에 설치된 한글 폰트 설정"""
    font_names = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP"]
    font_path = None
    for name in font_names:
        try:
            font_path = fm.findfont(fm.FontProperties(family=name))
            if font_path:
                plt.rcParams["font.family"] = name
                break
        except Exception:
            continue
    if not font_path:
        print("경고: 적절한 한글 폰트를 찾지 못했습니다. 기본 폰트로 출력됩니다.")


# ====== 보조 함수 ======
def clamp01(x):
    """값을 0.0과 1.0 사이로 제한합니다."""
    try:
        x_float = float(x)
    except (ValueError, TypeError):
        x_float = 0.0
    return max(0.0, min(1.0, x_float))


def calculate_indices(m):
    """JSON의 뇌파 수치(6개)를 바탕으로 5가지 복합 지표를 계산합니다."""
    stress = clamp01(m.get("stress", 0.0))
    engage = clamp01(m.get("engage", 0.0))
    relax = clamp01(m.get("relax", 0.0))
    excite = clamp01(m.get("excite", 0.0))
    interest = clamp01(m.get("interest", 0.0))
    focus = clamp01(m.get("focus", 0.0))

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


# ====== 실행 함수 ======
def run_rader_chart():
    """
    JSON 파일을 로드하여 막대 차트와 방사형 차트를 생성하고 이미지 파일로 저장합니다.
    """
    set_korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    # 1. JSON 파일 로드
    try:
        print(f"RaderChart: JSON 파일 로드 시도: {JSON_PATH.resolve()}")
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(
            f"RaderChart 오류: 입력 파일 '{JSON_PATH}'을 찾을 수 없습니다. (TxtToJson이 선행되어야 함)"
        )
        return False
    except json.JSONDecodeError:
        print(f"RaderChart 오류: {JSON_PATH} 파일의 JSON 형식이 올바르지 않습니다.")
        return False
    except Exception as e:
        print(f"RaderChart 오류: 파일 로드 중 오류 발생: {e}")
        return False

    # 2. 참가자 및 단계 데이터 추출
    try:
        participant_key = next(iter(data.keys()))
        steps_data = data[participant_key]["steps"]
    except (StopIteration, KeyError):
        print(
            "RaderChart 오류: JSON 파일에 참가자 데이터 또는 'steps' 데이터가 없습니다."
        )
        return False

    # 3. 모든 단계의 지표 계산
    all_step_indices = {}
    for step_name in STEPS_TO_PLOT:
        if step_name in steps_data:
            m = steps_data[step_name]
            all_step_indices[step_name] = calculate_indices(m)
        else:
            print(f"RaderChart 경고: {step_name} 데이터가 JSON 파일에 없어 건너뜁니다.")

    # 4. Bar Chart 생성 및 저장
    if all_step_indices:
        _create_bar_chart(all_step_indices, participant_key)

        # 5. Radar Chart (Step4) 생성 및 저장
        _create_radar_chart(all_step_indices.get("step4"), participant_key)

        return True
    else:
        print(
            "RaderChart 경고: 계산할 유효한 지표 데이터가 없어 차트 생성을 건너뜁니다."
        )
        return False


# ====== 내부 함수: Bar Chart 생성 ======
def _create_bar_chart(all_step_indices, participant_key):
    """막대 그래프를 생성하고 저장합니다."""

    # 데이터프레임으로 변환
    first_step_indices = next(iter(all_step_indices.values()))
    index_names = list(first_step_indices.keys())
    df = pd.DataFrame(all_step_indices).T
    df = df[index_names]

    # 플롯 준비
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(index_names))
    n_steps = len(STEPS_TO_PLOT)
    start_x = x - (BAR_WIDTH * (n_steps - 1) / 2)

    # 각 단계별로 막대 플롯
    for i, step_name in enumerate(STEPS_TO_PLOT):
        if step_name not in df.index:
            continue

        bar_x = start_x + (i * BAR_WIDTH)
        values_100 = df.loc[step_name].values * 100

        rects = ax.bar(
            bar_x, values_100, BAR_WIDTH, label=step_name, color=BAR_COLORS[i]
        )

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.0f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 차트 설정
    ax.set_ylabel("지표 점수", fontsize=12)
    ax.set_title(f"{participant_key} 단계별 지표 비교", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(index_names, fontsize=12)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", title="단계")
    plt.tight_layout()

    # 저장
    BAR_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(BAR_OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)  # 메모리 해제
    print(f"RaderChart: 막대 그래프 저장 완료: {BAR_OUT_PATH.resolve()}")


# ====== 내부 함수: Radar Chart 생성 ======
def _create_radar_chart(radar_indices, participant_key):
    """방사형 차트를 생성하고 저장합니다."""
    if not radar_indices:
        print("RaderChart 경고: step4 데이터가 없어 방사형 차트 생성을 건너뜁니다.")
        return

    labels = list(radar_indices.keys())
    values = [radar_indices[k] * 100 for k in labels]
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    values_closed = values + [values[0]]

    # 차트 설정
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.plot(
        angles_closed,
        values_closed,
        linewidth=3,
        linestyle="solid",
        color=RADAR_COLOR,
        label="step4 점수",
    )
    ax.fill(angles_closed, values_closed, color=RADAR_COLOR, alpha=0.3)

    # 축 설정
    ax.set_xticks(angles)
    ax.set_xticklabels([""] * N)
    y_ticks = np.arange(20, 101, 20)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y)}" for y in y_ticks], color="gray", size=10)
    ax.set_ylim(0, 100)

    # 데이터 레이블 표시
    DATA_LABEL_OFFSET = 5
    for angle, value in zip(angles, values):
        text_y = value + DATA_LABEL_OFFSET
        ax.text(
            angle,
            text_y,
            f"{int(round(value))}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
        )

    # 지표 레이블 표시
    TEXT_OFFSET = 12
    for angle, label in zip(angles, labels):
        ha_align = "center"
        if angle == np.pi:
            ha_align = "right"

        ax.text(
            angle,
            100 + TEXT_OFFSET,
            label,
            ha=ha_align,
            va="center",
            fontsize=12,
        )

    title = f"{participant_key}"
    ax.set_title(title, y=1.08, loc="left", fontsize=14)
    plt.tight_layout()

    # 저장
    RADAR_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(RADAR_OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)  # 메모리 해제
    print(f"RaderChart: 방사형 차트 저장 완료: {RADAR_OUT_PATH.resolve()}")


if __name__ == "__main__":
    run_rader_chart()

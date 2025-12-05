import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm


INPUT_DIR = Path("output/Emotion_EEG/Report_Json_Data")
OUTPUT_DIR = Path("output/Emotion_EEG/Chart_Result")

JSON_PATH = INPUT_DIR / "Report_Data.json"
# 파일명은 participant_name 기반으로 동적 생성됨


STEPS_TO_PLOT = ["step2", "step3", "step4"]
BAR_COLORS = ["#6BAED6", "#74C476", "#FD8D3C"]
RADAR_COLOR = "darkorange"
BAR_WIDTH = 0.25



def set_korean_font():
                           
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



def clamp01(x):
                                
    try:
        x_float = float(x)
    except (ValueError, TypeError):
        x_float = 0.0
    return max(0.0, min(1.0, x_float))


def calculate_indices(m):
                                                 
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

    
    for k in indices:
        indices[k] = float(np.clip(indices[k], 0.0, 1.0))

    return indices



def run_rader_chart(participant_id=None):
    """
    Radar Chart 생성
    participant_id: 특정 참가자 ID (None이면 첫 번째 참가자 사용)
    """
    set_korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    
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

    
    try:
        # participant_id가 지정된 경우 해당 참가자 찾기
        if participant_id:
            participant_key = None
            print(f"RaderChart: participant_id '{participant_id}'로 참가자 찾는 중...")
            print(f"  사용 가능한 키: {list(data.keys())}")
            
            for key in data.keys():
                # 키에서 이름 추출
                key_name = key.replace("participant_", "") if key.startswith("participant_") else key
                
                # 정확한 매칭 시도
                if key == participant_id or key == f"participant_{participant_id}":
                    participant_key = key
                    print(f"  ✅ 정확한 매칭 발견: {key}")
                    break
                
                # 이름으로 매칭 (participant_id가 이름인 경우, 예: "최준혁")
                if not participant_id.startswith("EB_"):
                    # 정확히 일치하는 경우
                    if participant_id == key_name:
                        participant_key = key
                        print(f"  ✅ 이름 매칭 발견: {key} (이름: {key_name})")
                        break
                    # 부분 일치 (더 느슨한 매칭)
                    elif participant_id in key_name or key_name in participant_id:
                        participant_key = key
                        print(f"  ✅ 부분 매칭 발견: {key} (이름: {key_name})")
                        break
                else:
                    # EB_ 형식인 경우 키에 포함되어 있는지 확인
                    if participant_id in key or key_name == participant_id:
                        participant_key = key
                        print(f"  ✅ EB_ 형식 매칭 발견: {key}")
                        break
            
            if not participant_key:
                # 찾지 못한 경우 첫 번째 참가자 사용
                participant_key = next(iter(data.keys()))
                print(f"  ⚠️ RaderChart 경고: {participant_id}에 해당하는 데이터를 찾지 못해 첫 번째 참가자 데이터를 사용합니다.")
                print(f"  사용된 키: {participant_key}")
            else:
                print(f"  ✅ 최종 선택된 키: {participant_key}")
        else:
            participant_key = next(iter(data.keys()))
            print(f"RaderChart: participant_id가 없어 첫 번째 참가자 사용: {participant_key}")
        
        steps_data = data[participant_key]["steps"]
        
        # participant_name 추출
        participant_name = participant_key.replace("participant_", "") if participant_key.startswith("participant_") else participant_key
    except (StopIteration, KeyError):
        print(
            "RaderChart 오류: JSON 파일에 참가자 데이터 또는 'steps' 데이터가 없습니다."
        )
        return False

    
    all_step_indices = {}
    for step_name in STEPS_TO_PLOT:
        if step_name in steps_data:
            m = steps_data[step_name]
            all_step_indices[step_name] = calculate_indices(m)
        else:
            print(f"RaderChart 경고: {step_name} 데이터가 JSON 파일에 없어 건너뜁니다.")

    
    if all_step_indices:
        _create_bar_chart(all_step_indices, participant_key, participant_name)

        
        _create_radar_chart(all_step_indices.get("step4"), participant_key, participant_name)

        return True
    else:
        print(
            "RaderChart 경고: 계산할 유효한 지표 데이터가 없어 차트 생성을 건너뜁니다."
        )
        return False



def _create_bar_chart(all_step_indices, participant_key, participant_name):
                             
    # participant_name 기반 파일 경로 생성
    bar_out_path = OUTPUT_DIR / f"{participant_name}_barchart.png"
    
    # 파일이 이미 존재하면 스킵
    if bar_out_path.exists():
        print(f"RaderChart: 막대 그래프 파일이 이미 존재합니다. 스킵: {bar_out_path.resolve()}")
        return

    
    first_step_indices = next(iter(all_step_indices.values()))
    index_names = list(first_step_indices.keys())
    df = pd.DataFrame(all_step_indices).T
    df = df[index_names]

    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(index_names))
    n_steps = len(STEPS_TO_PLOT)
    start_x = x - (BAR_WIDTH * (n_steps - 1) / 2)

    
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

    
    ax.set_ylabel("지표 점수", fontsize=12)
    ax.set_title(f"{participant_name} 단계별 지표 비교", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(index_names, fontsize=12)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", title="단계")
    plt.tight_layout()

    
    bar_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(bar_out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)  
    print(f"RaderChart: 막대 그래프 저장 완료: {bar_out_path.resolve()}")



def _create_radar_chart(radar_indices, participant_key, participant_name):
                             
    if not radar_indices:
        print("RaderChart 경고: step4 데이터가 없어 방사형 차트 생성을 건너뜁니다.")
        return

    # participant_name 기반 파일 경로 생성
    radar_out_path = OUTPUT_DIR / f"{participant_name}_radarchart.png"
    
    # 파일이 이미 존재하면 스킵
    if radar_out_path.exists():
        print(f"RaderChart: 방사형 차트 파일이 이미 존재합니다. 스킵: {radar_out_path.resolve()}")
        return

    labels = list(radar_indices.keys())
    values = [radar_indices[k] * 100 for k in labels]
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    values_closed = values + [values[0]]

    
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

    
    ax.set_xticks(angles)
    ax.set_xticklabels([""] * N)
    y_ticks = np.arange(20, 101, 20)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y)}" for y in y_ticks], color="gray", size=10)
    ax.set_ylim(0, 100)

    
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

    title = f"{participant_name}"
    ax.set_title(title, y=1.08, loc="left", fontsize=14)
    plt.tight_layout()

    
    radar_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(radar_out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)  
    print(f"RaderChart: 방사형 차트 저장 완료: {radar_out_path.resolve()}")


if __name__ == "__main__":
    run_rader_chart()

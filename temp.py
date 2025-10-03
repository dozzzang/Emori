import json
import re
from statistics import mean

# ---------- 유틸 ----------
def extract_line_value(label, text):
    m = re.search(rf"^{re.escape(label)}\s*:\s*(.*)$", text, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "") or ""


def extract_first(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""


def to_int100(x):
    try:
        v = float(x)
        if v <= 1.0:
            return int(round(v * 100))
        return int(round(v))
    except:
        return 0


def avg_or_zero(arr):
    return mean(arr) if arr else 0.0


def make_metrics(stress, relax, focus, excite, interest):
    return {
        "stress": to_int100(stress),
        "calm": to_int100(1.0 - stress if isinstance(stress, float) else 0),
        "creative": to_int100(excite),
        "focus": to_int100(focus),
        "relax": to_int100(relax),
        "comfort": to_int100(interest),
    }


# ---------- 원본 로드 ----------
with open("VR_Data.txt", "r", encoding="utf-8") as f:
    raw = f.read()

# ---------- 기본정보 ----------
name = extract_line_value("NAME", raw)
age = extract_line_value("AGE", raw)
gender = extract_line_value("GENDER", raw)
date = extract_line_value("REPORT_DAY", raw)  # 비어있을 수 있음

# ---------- GameData (+1 step 오프셋 반영) ----------
step1_emotion_color = extract_first(
    r"STEP1_EMOTION_COLOR[M]?\s*:\s*(.*)", raw
)  # → step2
step2_fill_rate = extract_first(r"STEP2_FILL_RATE\s*:\s*(.*)", raw)  # → step3
step3_fill_rate = extract_first(r"STEP3_FILL_RATE\s*:\s*(.*)", raw)  # → step4

# ---------- STEP.<State> 블록 파싱 ----------
# 예: ---------STEP.Breathe--------- / ---------STEP.Crosshair--------- / ---------STEP.Step1--------- ...
# 블록명과 블록본문을 모두 캡처
step_blocks = re.findall(
    r"-{5,}STEP\.([A-Za-z0-9_]+)-{5,}\s*(.*?)(?=(?:-{5,}STEP\.[A-Za-z0-9_]+-{5,})|$)",
    raw,
    flags=re.DOTALL,
)

# 상태별 PM 수치 누적
pm_by_state = {}  # state -> 리스트(dict)
for state, body in step_blocks:
    # 블록 안에서 PM_* 값 1세트만 존재할 수도, 여러 번 존재할 수도 있음
    # 일단 '해당 블록 내 첫 세트'를 사용(필요 시 여러 세트 평균도 가능)
    vals = {}
    for key in ["PM_Stress", "PM_Relax", "PM_Focus", "PM_Excite", "PM_Interest"]:
        m = re.search(rf"{key}\s*:\s*([0-9]*\.?[0-9]+)", body)
        if m:
            vals[key] = float(m.group(1))

    if vals:
        pm_by_state.setdefault(state, []).append(vals)


# 상태별 평균 → metrics
def metrics_of_state(state_name, fallback=None):
    arr = pm_by_state.get(state_name, [])
    if not arr:
        return fallback if fallback is not None else make_metrics(0, 0, 0, 0, 0)
    s = avg_or_zero([x.get("PM_Stress", 0.0) for x in arr])
    r = avg_or_zero([x.get("PM_Relax", 0.0) for x in arr])
    f = avg_or_zero([x.get("PM_Focus", 0.0) for x in arr])
    e = avg_or_zero([x.get("PM_Excite", 0.0) for x in arr])
    i = avg_or_zero([x.get("PM_Interest", 0.0) for x in arr])
    return make_metrics(s, r, f, e, i)


# ---------- 매핑 규칙 ----------
# 네가 주신 해석: Step1 == JSON.step2, Step2 == JSON.step3, Step3 == JSON.step4
m_step2 = metrics_of_state("Step1")
m_step3 = metrics_of_state("Step2")
m_step4 = metrics_of_state("Step3")

# ---------- 최종 JSON 구성 ----------
emotion_data = {
    f"participant_{name or 'unknown'}": {
        "basic_info": {"age": age, "gender": gender, "date": date},
        "steps": {
            "step2": {
                **m_step2,
                "emotion_color": step1_emotion_color,  # STEP1_EMOTION_COLOR → step2
            },
            "step3": {
                **m_step3,
                "fill_rate": step2_fill_rate,  # STEP2_FILL_RATE → step3
            },
            "step4": {
                **m_step4,
                "fill_rate": step3_fill_rate,  # STEP3_FILL_RATE → step4
            },
        },
    }
}

with open("Converted_Result.json", "w", encoding="utf-8") as f:
    json.dump(emotion_data, f, indent=4, ensure_ascii=False)

print("변환 완료! Converted_Result.json 파일이 생성되었습니다.")

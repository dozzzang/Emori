import json
import re
from statistics import mean
# ---------- 값 추출 by 정규표현식(라벨 : 값) ----------
def extract_line_value(label, text):
    m = re.search(rf"^{re.escape(label)}\s*:\s*([^\n])$", text, flags=re.MULTILINE)
    if m:
        extracted_value = m.group(1).strip()
        # 값이 공백이거나 비어있으면 'NULL' 반환
        return extracted_value if extracted_value else "NULL"
    else:
        # 라벨 자체를 찾지 못했으면 'NULL' 반환
        return "NULL"
# ---------- 감정 추출 ----------
def extract_emotion(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else "NULL"
# ---------- 파일 로드 ----------
with open("VR_Data.txt", "r", encoding="utf-8") as f:
    raw = f.read()

# ---------- UserInfo ----------
name = extract_line_value("NAME", raw)
age = extract_line_value("AGE", raw)
gender = extract_line_value("GENDER", raw)

# ---------- OwnerInfo ----------
date = extract_line_value("REPORT_DAY", raw)

# ---------- GameData ----------
step1_emotion_color = extract_emotion(r"STEP1_EMOTION_COLOR\s*:\s*(.*)", raw)  # step2
step2_fill_rate = extract_emotion(r"STEP2_FILL_RATE\s*:\s*(.*)", raw)  # step3
step3_fill_rate = extract_emotion(r"STEP3_FILL_RATE\s*:\s*(.*)", raw)  # step4

# ---------- STEP.<State> & 뇌파 수치 추출 ----------
step_blocks = re.findall(
    r"-{5,}STEP\.([A-Za-z0-9_]+)-{5,}\s*(.*?)(?=(?:-{5,}STEP\.[A-Za-z0-9_]+-{5,})|$)",
    raw,
    flags=re.DOTALL,
)

# Step별 PM 수치 누적
pm_by_state = {}
for state, body in step_blocks:
    if (
        state == "Breathe"
        or state == "Crosshair"
        or state == "PreStep1"
        or state == "PreStep2"
        or state == "PreStep3"
    ):
        continue

    print(body)

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

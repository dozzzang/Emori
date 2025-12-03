import json
import re
import glob
import os




DIR_PATH = "data/Emotion_EEG/VR_Result_Data"
OUTPUT_FILE_NAME = "output/Emotion_EEG/Augmented_Json_Data/Augmented_Report_Data.json"



def extract_line_value(label: str, text: str, null_value: str = "NULL") -> str:
           
    lab = label.strip()
    for line in text.splitlines():
        m = re.match(rf"^\s*{re.escape(lab)}\b\s*:\s*(.*)\s*$", line)
        if m:
            val = m.group(1).strip()
            return val if val else null_value
    return null_value


def extract_emotion(pattern: str, text: str, flags=0) -> str:
           
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else "NULL"



def make_metrics(stress, engage, relax, excite, interest, focus):
    return {
        "stress": stress,
        "engage": engage,
        "relax": relax,
        "excite": excite,
        "interest": interest,
        "focus": focus,
    }



def parse_step_blocks(raw: str):
           
    step_blocks = re.findall(
        r"-{5,}STEP\.([A-Za-z0-9_]+)-{5,}\s*(.*?)(?=(?:-{5,}STEP\.[A-Za-z0-9_]+-{5,})|$)",
        raw,
        flags=re.DOTALL,
    )

    pm_by_state = {}

    for state, body in step_blocks:
        
        if state.lower() in [
            "breathe",
            "crosshair",
            "prestep1",
            "prestep2",
            "prestep3",
        ]:
            continue

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




def parse_pm_step_summary(raw: str):
    summary_by_state = {}

    pattern = re.compile(
        r"PM_STEP_(\w+)\s*(.*?)(?=PM_STEP_\w+|---------End---------|$)",
        re.DOTALL,
    )

    for state, body in pattern.findall(raw):
        vals = {}
        for key in ["Stress", "Engage", "Relax", "Excite", "Interest", "Focus"]:
            m = re.search(
                rf"PM_AVERAGE_{key}\s*:\s*([0-9]*\.?[0-9]+)\s*%?",
                body,
            )
            if m:
                v = float(m.group(1))
                
                vals[f"PM_{key}"] = v / 100.0

        if vals:
            summary_by_state[state] = vals

    return summary_by_state



def metrics_of_state(
    state_name: str,
    pm_by_state: dict,
    pm_step_summary: dict | None = None,
):
           
    
    if pm_step_summary is not None and state_name in pm_step_summary:
        x = pm_step_summary[state_name]
        return make_metrics(
            x.get("PM_Stress", 0.0),
            x.get("PM_Engage", 0.0),
            x.get("PM_Relax", 0.0),
            x.get("PM_Excite", 0.0),
            x.get("PM_Interest", 0.0),
            x.get("PM_Focus", 0.0),
        )

    
    arr = pm_by_state.get(state_name, [])
    if not arr:
        return make_metrics(0, 0, 0, 0, 0, 0)

    x = arr[0]
    return make_metrics(
        x.get("PM_Stress", 0.0),
        x.get("PM_Engage", 0.0),
        x.get("PM_Relax", 0.0),
        x.get("PM_Excite", 0.0),
        x.get("PM_Interest", 0.0),
        x.get("PM_Focus", 0.0),
    )



def parse_single_txt(raw: str, source_path: str):
    
    name = extract_line_value("NAME", raw)
    age = extract_line_value("AGE", raw)
    gender = extract_line_value("GENDER", raw)
    date = extract_line_value("REPORT_DAY", raw)

    
    step1_emotion_color = extract_emotion(r"STEP1_EMOTION_COLOR\s*:\s*(.*)", raw)
    step2_fill_rate = extract_emotion(r"STEP2_FILL_RATE\s*:\s*(.*)", raw)
    step3_fill_rate = extract_emotion(r"STEP3_FILL_RATE\s*:\s*(.*)", raw)

    
    
    pm_by_state = parse_step_blocks(raw)
    
    
    pm_step_summary = parse_pm_step_summary(raw)
    

    
    
    
    
    m_step2 = metrics_of_state("Step1", pm_by_state, pm_step_summary)
    m_step3 = metrics_of_state("Step2", pm_by_state, pm_step_summary)
    m_step4 = metrics_of_state("Step3", pm_by_state, pm_step_summary)

    
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    participant_id = (
        f"participant_{name}" if name and name != "NULL" else f"participant_{base_name}"
    )

    
    participant_data = {
        "basic_info": {
            "age": age,
            "gender": gender,
            "date": date,
        },
        "steps": {
            "step2": {"emotion_color": step1_emotion_color, **m_step2},
            "step3": {"fill_rate": step2_fill_rate, **m_step3},
            "step4": {"fill_rate": step3_fill_rate, **m_step4},
        },
    }

    return participant_id, participant_data



def run_txt_to_json_all():
           
    
    file_pattern = os.path.join(DIR_PATH, "**", "RECORD*.txt")
    file_list = glob.glob(file_pattern, recursive=True)

    if not file_list:
        print(f"경로 '{DIR_PATH}'에서 RECORD*.txt 파일 없음")
        return False

    print(f"총 {len(file_list)}개의 RECORD*.txt 파일을 찾았습니다.")

    
    if os.path.exists(OUTPUT_FILE_NAME):
        try:
            with open(OUTPUT_FILE_NAME, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"기존 JSON 로드 완료: {OUTPUT_FILE_NAME}")
        except Exception as e:
            print(f"기존 JSON 로드 오류, 새로 생성: {e}")
            existing_data = {}
    else:
        existing_data = {}

    
    new_data = {}

    for path in sorted(file_list):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception as e:
            print(f"파일 읽기 오류 ({path}): {e}")
            continue

        if not raw:
            print(f"빈 파일 건너뜀: {path}")
            continue

        pid, pdata = parse_single_txt(raw, path)

        
        original_pid = pid
        idx = 2
        while pid in existing_data or pid in new_data:
            pid = f"{original_pid}_{idx}"
            idx += 1

        new_data[pid] = pdata

    
    merged = {**existing_data, **new_data}

    
    os.makedirs(os.path.dirname(OUTPUT_FILE_NAME), exist_ok=True)
    try:
        with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=4, ensure_ascii=False)
        print(f"JSON 병합 완료: 총 {len(merged)}명의 참가자가 저장됨")
        return True
    except Exception as e:
        print(f"JSON 저장 오류: {e}")
        return False


if __name__ == "__main__":
    run_txt_to_json_all()

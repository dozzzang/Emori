import json
import re
import glob
import os



DIR_PATH = "data/Emotion_EEG/VR_Result_Data"
OUTPUT_FILE_NAME = "output/Emotion_EEG/Report_Json_Data/Report_Data.json"



def extract_line_value(label: str, text: str, null_value="NULL") -> str:
    lab = label.strip()
    for line in text.splitlines():
        
        m = re.match(rf"^\s*{re.escape(lab)}\b\s*:\s*(.*)\s*$", line)
        if m:
            val = m.group(1).strip()
            return val if val else null_value
    return null_value



def extract_emotion(pattern, text, flags=0):
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



def metrics_of_state(state_name, pm_by_state, pm_step_summary=None, fallback=None):
    
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
        return fallback if fallback is not None else make_metrics(0, 0, 0, 0, 0, 0)

    x = arr[0]
    s = x.get("PM_Stress", 0.0)
    eg = x.get("PM_Engage", 0.0)
    r = x.get("PM_Relax", 0.0)
    ex = x.get("PM_Excite", 0.0)
    i = x.get("PM_Interest", 0.0)
    f = x.get("PM_Focus", 0.0)

    return make_metrics(s, eg, r, ex, i, f)



def run_txt_to_json():
                                              

    file_pattern = os.path.join(DIR_PATH, "RECORD*.txt")
    file_list = glob.glob(file_pattern)
    raw = None

    
    if not file_list:
        print(
            f"경로 '{DIR_PATH}'에서 'RECORD'로 시작하는 .txt 파일을 찾을 수 없습니다."
        )
        return False
    else:
        INPUT_FILE_NAME = file_list[0]
        try:
            with open(INPUT_FILE_NAME, "r", encoding="utf-8") as f:
                raw = f.read()
                print(f"TxtToJson: 성공적으로 파일을 로드했습니다: {INPUT_FILE_NAME}")
        except Exception as e:
            print(f"TxtToJson: 파일을 읽는 중 오류가 발생했습니다: {e}")
            return False

    
    if not raw:
        return False

    

    
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

    
    result_data = {
        f"participant_{name or 'unknown'}": {
            "basic_info": {"age": age, "gender": gender, "date": date},
            "steps": {
                "step2": {"emotion_color": step1_emotion_color, **m_step2},
                "step3": {"fill_rate": step2_fill_rate, **m_step3},
                "step4": {"fill_rate": step3_fill_rate, **m_step4},
            },
        }
    }

    
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE_NAME), exist_ok=True)
    except Exception:
        pass

    try:
        with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
            print(f"TxtToJson: JSON Data 생성 완료: {OUTPUT_FILE_NAME}")
            return True
    except Exception as e:
        print(f"TxtToJson: JSON 파일 저장 중 오류 발생: {e}")
        return False


if __name__ == "__main__":
    run_txt_to_json()

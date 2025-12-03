import glob
import json
import re
import random
import os
from typing import Dict, Any, Tuple
from pathlib import Path







DIR_PATH = "data/Emotion_EEG/VR_Result_Data"

file_pattern = os.path.join(DIR_PATH, "RECORD*.txt")
file_list = glob.glob(file_pattern)

if not file_list:
    raise FileNotFoundError(f"'{DIR_PATH}' 안에 RECORD*.txt 파일이 없습니다.")


BASE_INPUT_FILE = Path(file_list[0])
OUTPUT_JSON_FILE = Path(
    "output/Emotion_EEG/Augmented_Json_Data/Augmented_Report_Data.json"
)
EMOTION_CHOICES = ["Happy", "Fear", "Sad", "Surprise", "Angry", "Dislike"]
FILL_RATE_CHOICES_STEP2 = ["Highest", "High", "Low", "Least"]
FILL_RATE_CHOICES_STEP3 = ["Highest", "Higher", "Half", "Less", "Least"]
PM_AUGMENT_RANGE = 0.2  
PM_CORRELATION_NOISE = 0.2  
PM_NEAR_ZERO_THRESHOLD = 0.05  
PM_NEAR_ZERO_MAX = 0.1  



def extract_line_value(label: str, text: str, null_value=None) -> Any:
                             
    lab = label.strip()
    for line in text.splitlines():
        m = re.match(rf"^\s*{re.escape(lab)}\s*:\s*(.*)\s*$", line)
        if m:
            val = m.group(1).strip()
            return val if val else null_value
    return null_value


def extract_emotion(pattern, text, flags=0):
                                  
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def make_metrics(stress, engage, relax, excite, interest, focus):
                                        
    return {
        "stress": stress,
        "engage": engage,
        "relax": relax,
        "excite": excite,
        "interest": interest,
        "focus": focus,
    }


def extract_base_data(raw_content: str) -> Dict[str, Any]:
    data = {}

    
    data["NAME"] = extract_line_value("NAME", raw_content)
    data["AGE"] = extract_line_value("AGE", raw_content)
    data["GENDER"] = extract_line_value("GENDER", raw_content)
    data["REPORT_DAY"] = extract_line_value("REPORT_DAY", raw_content)

    
    data["STEP1_EMOTION_COLOR"] = extract_emotion(
        r"STEP1_EMOTION_COLOR\s*:\s*(.*)", raw_content
    )
    data["STEP2_FILL_RATE"] = extract_emotion(
        r"STEP2_FILL_RATE\s*:\s*(.*)", raw_content
    )
    data["STEP3_FILL_RATE"] = extract_emotion(
        r"STEP3_FILL_RATE\s*:\s*(.*)", raw_content
    )

    
    pm_by_state = {}
    target_states = ["Step1", "Step2", "Step3"]
    pm_keys_full = [
        "PM_Stress",
        "PM_Engage",
        "PM_Relax",
        "PM_Excite",
        "PM_Interest",
        "PM_Focus",
    ]

    
    step_blocks = re.findall(
        r"-{5,}STEP\.([A-Za-z0-9_]+)-{5,}\s*(.*?)(?=(?:-{5,}STEP\.[A-Za-z0-9_]+-{5,})|$)",
        raw_content,
        flags=re.DOTALL,
    )

    for state, body in step_blocks:
        if state not in target_states:
            continue

        vals = {}
        
        for key in pm_keys_full:
            m = re.search(rf"{key}\s*:\s*([0-9]*\.?[0-9]+)", body)
            if m:
                try:
                    vals[key] = float(m.group(1))
                except ValueError:
                    vals[key] = 0.0
            else:
                vals[key] = 0.0

        if vals and not pm_by_state.get(state):
            
            pm_by_state[state] = make_metrics(
                vals.get("PM_Stress", 0.0),
                vals.get("PM_Engage", 0.0),
                vals.get("PM_Relax", 0.0),
                vals.get("PM_Excite", 0.0),
                vals.get("PM_Interest", 0.0),
                vals.get("PM_Focus", 0.0),
            )

    data["PM_Step1"] = pm_by_state.get("Step1", make_metrics(0, 0, 0, 0, 0, 0))
    data["PM_Step2"] = pm_by_state.get("Step2", make_metrics(0, 0, 0, 0, 0, 0))
    data["PM_Step3"] = pm_by_state.get("Step3", make_metrics(0, 0, 0, 0, 0, 0))

    return data



def get_constrained_value(base_val, noise_factor=PM_CORRELATION_NOISE):
                                             
    min_val = max(0.001, base_val * (1 - noise_factor))
    max_val = min(1.0, base_val * (1 + noise_factor))
    return round(random.uniform(min_val, max_val), 7)



def augment_data(base_data: Dict[str, Any], index: int) -> Tuple[str, Dict[str, Any]]:
                                                   

    new_data = json.loads(json.dumps(base_data))

    
    base_file_id = os.path.basename(BASE_INPUT_FILE).replace(".txt", "")
    participant_id = f"participant_{base_file_id}_AUG{index+1:03d}"

    
    new_data["STEP1_EMOTION_COLOR"] = random.choice(EMOTION_CHOICES)
    new_data["STEP2_FILL_RATE"] = random.choice(FILL_RATE_CHOICES_STEP2)
    new_data["STEP3_FILL_RATE"] = random.choice(FILL_RATE_CHOICES_STEP3)

    
    for step_key in ["PM_Step1", "PM_Step2", "PM_Step3"]:

        if index >= 9:
            
            
            relax_base = random.uniform(0.1, 0.9)

            
            stress_base = 1.0 - relax_base

            
            
            cognitive_base = random.uniform(0.001, 1.0)

        pm_block = new_data[step_key]

        for key in pm_block.keys():

            if index < 9:
                
                original_value = pm_block.get(key, 0.5)

                
                noise_min = max(0.0, original_value * (1 - PM_AUGMENT_RANGE))
                noise_max = min(1.0, original_value * (1 + PM_AUGMENT_RANGE))

                
                if original_value < PM_NEAR_ZERO_THRESHOLD:
                    new_value = round(random.uniform(0.001, PM_NEAR_ZERO_MAX), 7)
                else:
                    new_value = round(random.uniform(noise_min, noise_max), 7)

            else:
                
                if key == "relax":
                    new_value = get_constrained_value(relax_base)
                elif key == "stress":
                    
                    new_value = get_constrained_value(stress_base)
                elif key in ["engage", "interest", "focus"]:
                    
                    new_value = get_constrained_value(cognitive_base)
                elif key == "excite":
                    
                    new_value = round(random.uniform(0.001, 1.0), 7)
                else:
                    
                    new_value = round(random.uniform(0.001, 1.0), 7)

            pm_block[key] = new_value

    
    participant_data = {
        "basic_info": {
            "age": new_data.get("AGE"),
            "gender": new_data.get("GENDER"),
            "date": new_data.get("REPORT_DAY"),
        },
        "steps": {
            "step2": {
                "emotion_color": new_data["STEP1_EMOTION_COLOR"],
                **new_data["PM_Step1"],
            },
            "step3": {
                "fill_rate": new_data["STEP2_FILL_RATE"],
                **new_data["PM_Step2"],
            },
            "step4": {
                "fill_rate": new_data["STEP3_FILL_RATE"],
                **new_data["PM_Step3"],
            },
        },
    }

    return participant_id, participant_data



def replace_none_with_null_string(data):
                                                 
    if isinstance(data, dict):
        return {k: replace_none_with_null_string(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_none_with_null_string(i) for i in data]
    elif data is None:
        return "NULL"
    else:
        return data



def main_augmentation_conversion(num_samples: int):
    
    if os.path.exists(OUTPUT_JSON_FILE):
        print(f"'{OUTPUT_JSON_FILE}' 파일이 이미 존재합니다. 데이터 증강은 건너뜁니다.")
        print("만약 데이터를 다시 증강하고 싶다면 파일을 삭제 후 실행하세요.")
        return

    random.seed(42)  

    
    try:
        with open(BASE_INPUT_FILE, "r", encoding="utf-8") as f:
            original_txt_content = f.read()
        print(f"'{BASE_INPUT_FILE}' 파일 내용을 성공적으로 읽었습니다.")
    except FileNotFoundError:
        print(
            f"오류: 증강에 사용할 원본 파일 '{BASE_INPUT_FILE}'을(를) 찾을 수 없습니다."
        )
        return
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
        return

    
    base_data = extract_base_data(original_txt_content)

    if not base_data:
        print(
            "원본 TXT 내용에서 필요한 기본 데이터를 추출하지 못했습니다. (NAME이나 STEP 정보 확인 필요)"
        )
        return

    consolidated_data = {}
    print(f"총 {num_samples}개의 증강 데이터를 생성합니다...")

    
    for i in range(num_samples):
        pid, data = augment_data(base_data, i)
        consolidated_data[pid] = data

    
    final_data = replace_none_with_null_string(consolidated_data)

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(
        f"증강 완료! 총 {num_samples}개의 데이터 세트가 '{OUTPUT_JSON_FILE}'로 생성되었습니다."
    )


if __name__ == "__main__":
    
    main_augmentation_conversion(num_samples=50)

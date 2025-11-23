import json
import re
import glob
import os

# 추가 제공받은 txt 데이터 모두를 Json으로 변환하는 코드

# ---------- 상수 정의 ----------
DIR_PATH = "data/Emotion_EEG/VR_Result_Data"
OUTPUT_FILE_NAME = "output/Emotion_EEG/Augmented_Json_Data/Augmented_Report_Data.json"


# ---------- 값 추출 by 정규표현식(라벨 : 값) ----------
def extract_line_value(label: str, text: str, null_value: str = "NULL") -> str:
    """
    NAME, AGE, REPORT_DAY 등의 '라벨 : 값' 형태에서 값을 한 줄 단위로 추출.
    """
    lab = label.strip()
    for line in text.splitlines():
        m = re.match(rf"^\s*{re.escape(lab)}\b\s*:\s*(.*)\s*$", line)
        if m:
            val = m.group(1).strip()
            return val if val else null_value
    return null_value


def extract_emotion(pattern: str, text: str, flags=0) -> str:
    """
    STEP1_EMOTION_COLOR 같은 특정 패턴에서 문자열 전체를 추출.
    """
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else "NULL"


# ---------- 공통 메트릭 구조 ----------
def make_metrics(stress, engage, relax, excite, interest, focus):
    return {
        "stress": stress,
        "engage": engage,
        "relax": relax,
        "excite": excite,
        "interest": interest,
        "focus": focus,
    }


# ---------- (구버전) STEP.Step1/2/3 블록에서 PM_* 추출 ----------
def parse_step_blocks(raw: str):
    """
    ---------STEP.Step1--------- ~ 다음 STEP 또는 파일 끝까지를 하나의 블록으로 보고
    그 안에서 PM_Stress, PM_Engage, ... 등을 찾는다.

    구버전(20250515 형식)의
    - STEP.Breathe
    - STEP.PreStep*
    - STEP.Step1/2/3
    등을 처리하기 위한 로직.
    """
    step_blocks = re.findall(
        r"-{5,}STEP\.([A-Za-z0-9_]+)-{5,}\s*(.*?)(?=(?:-{5,}STEP\.[A-Za-z0-9_]+-{5,})|$)",
        raw,
        flags=re.DOTALL,
    )

    pm_by_state = {}

    for state, body in step_blocks:
        # 숨고르기(Breathe)나 크로스헤어, PreStep* 단계는 EEG 요약에서 제외
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
            # ex) PM_Stress : 0.12345
            m = re.search(rf"{key}\s*:\s*([0-9]*\.?[0-9]+)", body)
            if m:
                vals[key] = float(m.group(1))

        if vals:
            pm_by_state.setdefault(state, []).append(vals)

    return pm_by_state


# ---------- (신버전) PM_STEP_Step1/2/3 요약 블록에서 PM_AVERAGE_* 추출 ----------
# 새로운 형식의 데이터에 대한 처리를 위하여 PM_AVERAGE_* 에 대하여 추출하는 로직 추가 (PM_Stress에 대한 수치만 존재하고, AVG 형태로 다른 값들에 대하여 표시되어 있어 해당 방식으로 값 추출)
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
                # 0~100% -> 0~1 로 변환
                vals[f"PM_{key}"] = v / 100.0

        if vals:
            summary_by_state[state] = vals

    return summary_by_state


# ---------- 최종 메트릭 선택 로직 ----------
def metrics_of_state(
    state_name: str,
    pm_by_state: dict,
    pm_step_summary: dict | None = None,
):
    """
    1순위: PM_STEP_Step* 요약 블록에서 나온 값 사용 (신규 형식)
    2순위: STEP.Step* 블록에서 첫 번째로 등장한 값 사용 (구 형식)
    둘 다 없으면 0으로 채운 메트릭 반환.
    """
    # 1) 신버전 요약 블록 우선
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

    # 2) 구버전 STEP.Step* 블록
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


# ---------- 개별 TXT 파일 -> participant JSON ----------
def parse_single_txt(raw: str, source_path: str):
    # 1) 기본 정보
    name = extract_line_value("NAME", raw)
    age = extract_line_value("AGE", raw)
    gender = extract_line_value("GENDER", raw)
    date = extract_line_value("REPORT_DAY", raw)

    # 2) 감정 정보 (step1 emotion_color, step2/3 fill_rate)
    step1_emotion_color = extract_emotion(r"STEP1_EMOTION_COLOR\s*:\s*(.*)", raw)
    step2_fill_rate = extract_emotion(r"STEP2_FILL_RATE\s*:\s*(.*)", raw)
    step3_fill_rate = extract_emotion(r"STEP3_FILL_RATE\s*:\s*(.*)", raw)

    # 3) EEG 관련 수치 파싱
    # 3-1) 구형 STEP.Step* 블록
    pm_by_state = parse_step_blocks(raw)
    # print("pm_step_summary =", pm_by_state)
    # 3-2) 신형 PM_STEP_Step* 요약 블록
    pm_step_summary = parse_pm_step_summary(raw)
    # print("pm_step_summary =", pm_step_summary)

    # 4) step2/3/4 에 대응되는 EEG 메트릭 선택
    #    - step2  <- Step1
    #    - step3  <- Step2
    #    - step4  <- Step3
    m_step2 = metrics_of_state("Step1", pm_by_state, pm_step_summary)
    m_step3 = metrics_of_state("Step2", pm_by_state, pm_step_summary)
    m_step4 = metrics_of_state("Step3", pm_by_state, pm_step_summary)

    # 5) participant ID 구성
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    participant_id = (
        f"participant_{name}" if name and name != "NULL" else f"participant_{base_name}"
    )

    # 6) 최종 participant 데이터 구조
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


# ---------- 여러 TXT 파일을 한 번에 읽어 기존 JSON에 병합 ----------
def run_txt_to_json_all():
    """
    DIR_PATH 아래의 모든 RECORD*.txt 파일을 찾아서
    - 개별 participant JSON으로 파싱
    - 기존 OUTPUT_FILE_NAME(Augmented_Report_Data.json)이 있으면 이어 붙이고
    - 없으면 새로 만든다.
    """
    # 기존의 데이터 부족으로 인한 자체 증강한 json 데이터 뒤에 이어서 저장
    file_pattern = os.path.join(DIR_PATH, "**", "RECORD*.txt")
    file_list = glob.glob(file_pattern, recursive=True)

    if not file_list:
        print(f"경로 '{DIR_PATH}'에서 RECORD*.txt 파일 없음")
        return False

    print(f"총 {len(file_list)}개의 RECORD*.txt 파일을 찾았습니다.")

    # 1) 기존 JSON 불러오기
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

    # 2) 이번에 새로 파싱한 데이터
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

        # 기존/신규 모두에서 participant ID 중복 방지
        original_pid = pid
        idx = 2
        while pid in existing_data or pid in new_data:
            pid = f"{original_pid}_{idx}"
            idx += 1

        new_data[pid] = pdata

    # 3) 기존 + 신규 병합
    merged = {**existing_data, **new_data}

    # 4) 저장
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

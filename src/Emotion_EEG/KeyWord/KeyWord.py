# --------------------------------------------------------
# 키워드 기준: 1. 감정, 2. valence(긍/부정), 3. arousal(활성도),
# 4. 몰입 * 집중(인지 효율, 과제 수행의 효율성 및 몰입의 깊이 판단)
# --------------------------------------------------------

import json
from pathlib import Path
import math

# ====== Path & Configuration ======
INPUT_DIR = Path("output/Emotion_EEG/Report_Json_Data")
JSON_PATH = INPUT_DIR / "Report_Data.json"

# -------- 감정 세분화 사전(4단계) --------
EMOTION_INTENSITY = {
    "Happy": {"Full": "신남", "High": "기쁨", "Half": "편안", "Low": "만족"},
    "Surprise": {"Full": "충격", "High": "놀람", "Half": "긴장", "Low": "어이없는"},
    "Sad": {"Full": "절망", "High": "슬픔", "Half": "걱정", "Low": "속상"},
    "Fear": {"Full": "공포", "High": "두려움", "Half": "겁나는", "Low": "불안"},
    "Angry": {"Full": "분노가득", "High": "화남", "Half": "짜증", "Low": "섭섭함"},
    "Disgust": {
        "Full": "역겨움",
        "High": "너무싫음",
        "Half": "싫증남",
        "Low": "지겨움",
    },
}

# -------- 임계값(절대 기준) --------
VALENCE_HIGH, VALENCE_LOW = 0.60, 0.40
AROUSAL_HIGH, AROUSAL_LOW = 0.65, 0.35
TE_HIGH, TE_LOW = 0.70, 0.40

# -------- step별 가중치 설정(후반의 데이터에 더 높은 가중치 부여) --------
STEP_WEIGHTS = {"step2": 0.2, "step3": 0.3, "step4": 0.5}


# =================================================================
# 보조 함수
# =================================================================


def emotion_tag_from_step2_step3(step2: dict, step3: dict) -> str:
    """step2의 감정 기본색과 step3의 채우기 비율을 조합하여 감정 태그 생성"""
    base = (step2.get("emotion_color") or "").strip()
    fr = (step3.get("fill_rate") or "").strip()
    label = EMOTION_INTENSITY.get(base, {}).get(fr)
    if not label:
        label = base if base else "미정"
    return f"#감정_{label}"


def valence_tag(value: float) -> str:
    """Valence(긍정/부정) 수치를 태그로 변환"""
    if value >= VALENCE_HIGH:
        return "#정서_긍정"
    if value <= VALENCE_LOW:
        return "#정서_부정"
    return "#정서_평온"


def arousal_tag(value: float) -> str:
    """Arousal(활성도) 수치를 태그로 변환"""
    if value >= AROUSAL_HIGH:
        return "#활성_높음"
    if value <= AROUSAL_LOW:
        return "#활성_낮음"
    return "#활성_보통"


def te_tag(value: float) -> str:
    """Engagement * Focus (몰입/집중) 효율성 수치를 태그로 변환"""
    if value >= TE_HIGH:
        return "#몰입집중_강함"
    if value <= TE_LOW:
        return "#몰입집중_약함"
    return "#몰입집중_보통"


def weighted_metric(steps: dict, func) -> float:
    """step2~4 값을 func(st)로 계산한 뒤 가중 평균 반환"""
    total, weight_sum = 0.0, 0.0
    for step_name, w in STEP_WEIGHTS.items():
        if step_name in steps:
            st = steps[step_name]
            # 모든 뇌파 지표가 유효한지 확인하는 로직 추가 (필요시)
            total += func(st) * w
            weight_sum += w
    return total / weight_sum if weight_sum > 0 else 0.0


def keywords_from_json(path: Path) -> list[tuple[str, list[str]]]:
    """
    JSON 파일 경로를 받아 참가자별 키워드 태그 리스트를 반환하는 핵심 함수
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"KeyWord 오류: 입력 JSON 파일 '{path}'을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError as e:
        print(f"KeyWord 오류: JSON 파일 디코딩 오류: {e}")
        return []

    participants = []
    # 참가자 데이터 구조 파악 및 목록화
    if isinstance(data, dict) and "steps" in data:
        # 단일 참가자 구조 (이름이 key가 아닌 경우)
        participants.append(("participant_NULL", data))
    elif isinstance(data, dict):
        # 다중 참가자 구조 (pid: data)
        for k, v in data.items():
            if isinstance(v, dict) and "steps" in v:
                participants.append((k, v))

    outputs = []
    for pid, pobj in participants:
        steps = pobj["steps"]

        # 1. 감정 태그
        step2 = steps.get("step2", {})
        step3 = steps.get("step3", {})
        emo_tag = emotion_tag_from_step2_step3(step2, step3)

        # --- Valence, Arousal, TE 가중평균 계산 ---
        # Valence (긍정성/부정성)
        def calc_valence(st):
            # 긍정 요소(excite, interest, engage) - 부정 요소(stress) / 4.0
            return (
                st.get("excite", 0.0)
                + st.get("interest", 0.0)
                + st.get("engage", 0.0)
                - st.get("stress", 0.0)
            ) / 4.0

        # Arousal (활성/각성)
        def calc_arousal(st):
            # excite, engage, focus 평균에서 relax를 감점 보정 역할
            return (
                st.get("excite", 0.0)
                + st.get("engage", 0.0)
                + st.get("focus", 0.0)
                - st.get("relax", 0.0)
            ) / 3.0

        # 몰입-집중 (인지 효율)
        def calc_te(st):
            # engage와 focus의 기하 평균 (둘 중 하나라도 낮으면 값이 낮아짐)
            return math.sqrt(
                max(0.0, st.get("engage", 0.0)) * max(0.0, st.get("focus", 0.0))
            )

        v_value = weighted_metric(steps, calc_valence)
        a_value = weighted_metric(steps, calc_arousal)
        te_value = weighted_metric(steps, calc_te)

        # 2. 태그 변환
        v_tag = valence_tag(v_value)
        a_tag = arousal_tag(a_value)
        te_tag_ = te_tag(te_value)

        outputs.append((pid, [emo_tag, v_tag, a_tag, te_tag_]))
    return outputs


# 실행 함수
def run_keyword_analysis():
    """키워드 분석을 실행하고 결과를 콘솔에 출력합니다."""
    print("KeyWord: 분석 시작...")
    results = keywords_from_json(JSON_PATH)

    if results:
        print("\n=== KeyWord 분석 결과 ===")
        for pid, tags in results:
            print(f"[{pid}] " + " ".join(tags))
        return True
    else:
        print("KeyWord: 분석할 유효한 참가자 데이터가 없습니다.")
        return False


if __name__ == "__main__":
    run_keyword_analysis()

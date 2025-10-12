# --------------------------------------------------------
# 키워드 기준: 1. 감정, 2. valence(긍/부정), 3. arousal(활성도),
# 4. 몰입 * 집중(인지 효율, 과제 수행의 효율성 및 몰입의 깊이 판단)
# --------------------------------------------------------

import json
from pathlib import Path
import math

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


def emotion_tag_from_step2_step3(step2: dict, step3: dict) -> str:
    base = (step2.get("emotion_color") or "").strip()
    fr = (step3.get("fill_rate") or "").strip()
    label = EMOTION_INTENSITY.get(base, {}).get(fr)
    if not label:
        label = base if base else "미정"
    return f"#감정_{label}"


def valence_tag(value: float) -> str:
    if value >= VALENCE_HIGH:
        return "#정서_긍정"
    if value <= VALENCE_LOW:
        return "#정서_부정"
    return "#정서_평온"


def arousal_tag(value: float) -> str:
    if value >= AROUSAL_HIGH:
        return "#활성_높음"
    if value <= AROUSAL_LOW:
        return "#활성_낮음"
    return "#활성_보통"


def te_tag(value: float) -> str:
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
            total += func(st) * w
            weight_sum += w
    return total / weight_sum if weight_sum > 0 else 0.0


def keywords_from_json(path: str) -> list[tuple[str, list[str]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    participants = []
    # 참가자 이름이 없는 경우
    if isinstance(data, dict) and "steps" in data:
        participants.append(("participant_NULL", data))
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and "steps" in v:
                participants.append((k, v))

    outputs = []
    for pid, pobj in participants:
        steps = pobj["steps"]

        # 감정: step2.base + step3.fill_rate
        step2 = steps.get("step2", {})
        step3 = steps.get("step3", {})
        emo_tag = emotion_tag_from_step2_step3(step2, step3)

        # --- Valence, Arousal, TE 가중평균 계산 ---
        def calc_valence(st):
            return (st["excite"] + st["interest"] + st["engage"] - st["stress"]) / 4.0

        # 분모 3 -> excite, engage, focus의 평균에서 relax를 감점 보정 역할로 처리
        def calc_arousal(st):
            return (st["excite"] + st["engage"] + st["focus"] - st["relax"]) / 3.0

        # 몰입-집중 -> 둘 중 하나라도 낮으면 효율성이 낮다고 판단
        def calc_te(st):
            return math.sqrt(max(0.0, st["engage"]) * max(0.0, st["focus"]))

        v_value = weighted_metric(steps, calc_valence)
        a_value = weighted_metric(steps, calc_arousal)
        te_value = weighted_metric(steps, calc_te)

        v_tag = valence_tag(v_value)
        a_tag = arousal_tag(a_value)
        te_tag_ = te_tag(te_value)

        outputs.append((pid, [emo_tag, v_tag, a_tag, te_tag_]))
    return outputs


if __name__ == "__main__":
    results = keywords_from_json(JSON_PATH)

    if results:
        print("\n=== 분석 결과 ===")
        for pid, tags in results:
            print(f"[{pid}] " + " ".join(tags))
    else:
        print("분석할 유효한 참가자 데이터가 없습니다.")

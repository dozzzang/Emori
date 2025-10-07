# --------------------------------------------------------
# 키워드 기준: 1. 감정, 2. valence(긍/부정), 3. arousal(활성도),
# 4. 몰입 * 집중(인지 효율, 과제 수행의 효율성 및 몰입의 깊이 판단)
# --------------------------------------------------------

import json
from pathlib import Path
import math

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



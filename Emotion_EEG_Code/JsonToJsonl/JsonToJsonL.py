import json
import pandas as pd

# ===== 1) 용어/사전 =====

EMOTION_KO = {
    "Happy": "기쁜",
    "Fear": "두려운",
    "Sad": "슬픈",
    "Surprise": "놀란",
    "Angry": "화나는",
    "Disgust": "싫은",
}

HAPPY_QUALITY = {"Low": "만족", "Half": "편안한", "High": "기쁜", "Full": "신나는"}
FEAR_QUALITY = {
    "Low": "불안한",
    "Half": "겁나는",
    "High": "두려운",
    "Full": "공포스러운",
}
SAD_QUALITY = {"Low": "속상한", "Half": "걱정되는", "High": "슬픈", "Full": "절망적인"}
SURPRISE_QUALITY = {
    "Low": "어이가 없는",
    "Half": "긴장되는",
    "High": "깜짝 놀란",
    "Full": "놀라 충격적인",
}
ANGRY_QUALITY = {
    "Low": "섭섭한",
    "Half": "짜증나는",
    "High": "화나는",
    "Full": "분노로 가득찬",
}
DISGUST_QUALITY = {
    "Low": "지겨운",
    "Half": "싫증나는",
    "High": "너무 싫은",
    "Full": "역겨운",
}

BODY_FEEL = {
    "Minimal": "발목까지 올라오는",
    "Low": "무릎까지 올라오는",
    "Half": "허리까지 올라오는",
    "High": "가슴까지 차오르는",
    "Full": "온몸을 가득 채운",
}

LABEL = {
    "focus": "집중",
    "interest": "흥미",
    "engage": "관여도",
    "excite": "각성",
    "relax": "이완",
    "stress": "스트레스",
}

import os
import re
import json
from pathlib import Path

# 파일 경로 설정
EMOTION_INPUT_DIR = 'data/emotionResult'
OUTPUT_DIR = 'output/emotionRelation/mainEmotion'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_main_emotion(emotion_filename: str):
    """
    emotionResult 파일에서 STEP1_EMOTION_COLOR 값을 추출하고 JSON으로 저장합니다.
    """
    input_path = os.path.join(EMOTION_INPUT_DIR, emotion_filename)
    
    if not os.path.exists(input_path):
        print(f"🛑 파일이 존재하지 않습니다: {input_path}")
        return None

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"🛑 파일 읽기 실패: {e}")
        return None

    # 정규 표현식을 사용하여 'STEP1_EMOTION_COLOR' 항목의 값 추출
    # 값 앞뒤의 공백을 제거하여 순수한 감정 단어만 얻습니다.
    match = re.search(r'STEP1_EMOTION_COLOR\s*:\s*(\w+)', content)
    
    if match:
        main_emotion = match.group(1).strip()
    else:
        main_emotion = "Unknown"  # 해당 항목을 찾지 못한 경우

    # 1단계 출력 데이터 형식
    output_data = {
      "filename": Path(emotion_filename).stem,
      "main_emotion": main_emotion, 
      "main_dot_color": "Black",
      "source_file": emotion_filename
    }

    # JSON 파일 저장
    output_filename = Path(emotion_filename).stem + '_mainEmotion.json'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 1단계 메인 감정 추출 완료: {output_path} (감정: {main_emotion})")
        return output_data
    except Exception as e:
        print(f"🛑 1단계 결과 저장 실패: {e}")
        return None

# # 실행 예시 (사용자가 파일을 지정한다고 가정)
# # sample_input_filename = "EB_001_emotionResult.txt" 
# # extract_main_emotion(sample_input_filename)
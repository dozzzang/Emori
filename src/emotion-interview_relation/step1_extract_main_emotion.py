import os
import re
from pathlib import Path


EMOTION_INPUT_DIR = 'data/emotionResult' 
OUTPUT_DIR = 'output/emotionRelation/mainEmotion' 


def extract_main_emotion_from_file(emotion_filepath: str, emotion_filename: str) -> str | None:
           
    
    if not os.path.exists(emotion_filepath):
        print(f"🛑 파일이 존재하지 않습니다: {emotion_filepath}")
        return None

    try:
        with open(emotion_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"🛑 파일 읽기 실패: {e}")
        return None

    
    
    match = re.search(r'STEP1_EMOTION_COLOR\s*:\s*(\w+)', content)
    
    if match:
        return match.group(1).strip()
    else:
        return "Unknown" 

def process_single_file(filename: str):
                                    
    input_path = os.path.join(EMOTION_INPUT_DIR, filename)
    main_emotion = extract_main_emotion_from_file(input_path, filename)

    if main_emotion:
        
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(main_emotion)
            print(f"✅ 메인 감정 추출 및 저장 완료: {output_path} (감정: **{main_emotion}**)")
        except Exception as e:
            print(f"🛑 결과 파일 저장 실패: {e}")

def process_all_files():
                               
    
    emotion_files = [f for f in os.listdir(EMOTION_INPUT_DIR) if f.endswith('.txt')]
    
    if not emotion_files:
        print(f"🛑 {EMOTION_INPUT_DIR} 폴더에 .txt 파일이 없습니다.")
        return

    print(f"📁 총 {len(emotion_files)}개의 파일을 처리합니다.")
    for filename in emotion_files:
        process_single_file(filename)

def main():
                                             
    
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    print("="*40)
    print("🎯 1단계: 메인 감정 추출 시작")
    print(f"  > 소스 디렉토리: {EMOTION_INPUT_DIR}")
    print(f"  > 출력 디렉토리: {OUTPUT_DIR}")
    print("="*40)
    
    while True:
        choice = input("\n분석 방식을 선택하세요 (1: 단일 파일, 2: 모든 파일, Q: 종료): ")
        
        if choice == '1':
            filename = input("분석할 파일명을 입력하세요 (예: EB_001_emotionResult.txt): ")
            process_single_file(filename)
        elif choice == '2':
            process_all_files()
            break 
        elif choice.upper() == 'Q':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다. 1, 2 또는 Q를 입력하세요.")

if __name__ == "__main__":
    main()
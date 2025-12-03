import os
import re
import glob
from pathlib import Path


EEG_INPUT_DIR = 'data/Emotion_EEG/VR_Result_Data'
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

def find_eeg_file_by_participant_id(participant_id: str):
    file_pattern = os.path.join(EEG_INPUT_DIR, "RECORD*.txt")
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        return None
    
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            name_match = re.search(r'NAME\s*:\s*(\S+)', content)
            if name_match:
                name = name_match.group(1).strip()
                if participant_id in name or name in participant_id:
                    return file_path
            
            base_name = Path(file_path).stem
            if participant_id in base_name:
                return file_path
        except Exception as e:
            continue
    
    if len(file_list) == 1:
        return file_list[0]
    
    return None

def process_single_file(participant_id: str):
    eeg_file_path = find_eeg_file_by_participant_id(participant_id)
    
    if not eeg_file_path:
        print(f"🛑 {participant_id}에 해당하는 뇌파 파일을 찾을 수 없습니다.")
        print(f"   검색 경로: {EEG_INPUT_DIR}/RECORD*.txt")
        return
    
    print(f"📂 뇌파 파일 발견: {eeg_file_path}")
    main_emotion = extract_main_emotion_from_file(eeg_file_path, participant_id)

    if main_emotion:
        output_filename = f"{participant_id}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(main_emotion)
            print(f"✅ 메인 감정 추출 및 저장 완료: {output_path} (감정: {main_emotion})")
        except Exception as e:
            print(f"🛑 결과 파일 저장 실패: {e}")

def process_all_files():
    file_pattern = os.path.join(EEG_INPUT_DIR, "RECORD*.txt")
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"🛑 {EEG_INPUT_DIR} 폴더에 RECORD*.txt 파일이 없습니다.")
        return

    print(f"📁 총 {len(file_list)}개의 뇌파 파일을 처리합니다.")
    for file_path in file_list:
        base_name = Path(file_path).stem
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            name_match = re.search(r'NAME\s*:\s*(\S+)', content)
            participant_id = name_match.group(1).strip() if name_match else base_name
            process_single_file(participant_id)
        except Exception as e:
            print(f"⚠️ 파일 처리 실패 ({file_path}): {e}")

def main():
                                             
    
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    print("="*40)
    print("🎯 1단계: 메인 감정 추출 시작")
    print(f"  > 소스 디렉토리: {EEG_INPUT_DIR}")
    print(f"  > 출력 디렉토리: {OUTPUT_DIR}")
    print("="*40)
    
    while True:
        choice = input("\n분석 방식을 선택하세요 (1: 단일 파일, 2: 모든 파일, Q: 종료): ")
        
        if choice == '1':
            participant_id = input("참가자 ID를 입력하세요 (예: EB_002): ")
            process_single_file(participant_id)
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
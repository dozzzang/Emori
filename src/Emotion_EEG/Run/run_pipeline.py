import sys
from pathlib import Path

# 파이썬이 src 폴더를 패키지의 루트로 인식하도록 시스템 경로 설정
# 현재 스크립트의 부모 디렉토리(Run)의 부모 디렉토리(Emotion_EEG)의 부모 디렉토리(src)를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))


# ----------------------------------------------------
# 다른 모듈 임포트
# ----------------------------------------------------

# (1) TxtToJson
from Emotion_EEG.TxtToJson import TxtToJson

# (2) RaderChart
from Emotion_EEG.Rader_Chart import RaderChart

# (3) KeyWord
from Emotion_EEG.KeyWord import KeyWord

# (4) JsonToJsonlMain
from Emotion_EEG.JsonToJsonl import JsonToJsonlMain

# (5) Llama3Main
from Emotion_EEG.DescriptiveSummary_Llama3 import Llama3Main

# ----------------------------------------------------


def run_all_pipeline():
    # 1. TxtToJson 실행
    TxtToJson.run_txt_to_json()

    # 2. RaderChart 실행
    RaderChart.run_rader_chart()

    # 3. KeyWord 실행
    KeyWord.run_keyword_analysis()

    # 4. JsonToJsonlMain 실행
    JsonToJsonlMain.run_json_to_jsonl()

    # 5. Llama3Main 실행
    Llama3Main.run_llama_inference()


if __name__ == "__main__":
    run_all_pipeline()

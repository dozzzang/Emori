import sys
from pathlib import Path



sys.path.append(str(Path(__file__).parent.parent.parent))







from Emotion_EEG.TxtToJson import TxtToJson


from Emotion_EEG.Rader_Chart import RaderChart


from Emotion_EEG.KeyWord import KeyWord


from Emotion_EEG.JsonToJsonl import JsonToJsonlMain


from Emotion_EEG.DescriptiveSummary_Llama3 import Llama3Main




def run_all_pipeline():
    
    TxtToJson.run_txt_to_json()

    
    RaderChart.run_rader_chart()

    
    KeyWord.run_keyword_analysis()

    
    JsonToJsonlMain.run_json_to_jsonl()

    
    Llama3Main.run_llama_inference()


if __name__ == "__main__":
    run_all_pipeline()

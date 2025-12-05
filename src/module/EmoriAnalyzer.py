

import os
import json
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq


current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

class EmoriAnalyzer:
    def __init__(self, eeg_json_path, llama_json_path):
        # 프로젝트 루트에서 .env 파일 로드
        env_path = current_dir.parent.parent / ".env"
        load_dotenv(dotenv_path=env_path)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        
        if not self.client:
            print("⚠️ [Analyzer] 경고: GROQ_API_KEY가 없습니다. API 점수 계산이 불가능합니다.")

        self.eeg_data = self._load_json(eeg_json_path)
        self.llama_data = self._load_json(llama_json_path)

    def _load_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ [Analyzer] 파일 로드 실패 ({path}): {e}")
            return {}

    def get_score_from_api(self):
                   
        analysis_data = self.llama_data.get("analysis_result", [])
        if not analysis_data:
            print("ℹ️ [Analyzer] 감정 데이터가 비어있습니다. (기본값 0.5 반환)")
            return 0.5

        data_str = json.dumps(analysis_data, ensure_ascii=False)

        if not self.client:
            return 0.5

        prompt = f"""
        ### Role
        You are an expert Clinical Psychologist.

        ### Task
        Analyze the following JSON list of emotions and intensities extracted from a student's counseling session.
        Calculate a comprehensive "Verbal Positivity Score" (integer from 0 to 100).
        
        ### Input Data
        {data_str}

        ### Output Format (JSON ONLY)
        {{
            "score": <int>,
            "reason": "<short explanation in Korean>"
        }} 
        """
        
        try:
            print(f"🤖 [Analyzer] Groq API로 감정 점수 분석 요청 중...")
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, 
                temperature=0.3
            )
            
            result = json.loads(completion.choices[0].message.content)
            
            if "analysis_result" in result:
                score = result["analysis_result"].get("score", 50)
            else:
                score = result.get("score", 50)
            
            reason = result.get("reason", "분석 완료")
            print(f"✅ [Analyzer] 분석 완료: {score}점 - {reason}")
            
            return float(score) / 100.0

        except Exception as e:
            print(f"❌ [Analyzer] API 호출 오류: {e}")
            return 0.5

    def analyze(self, participant_id="participant_1"):
                                                  
        
        try:
            if participant_id not in self.eeg_data and self.eeg_data:
                participant_id = next(iter(self.eeg_data))
            
            steps = self.eeg_data.get(participant_id, {}).get('steps', {})
            vr_final_state = steps.get('step4', {}) 
        except:
            print("❌ [Analyzer] EEG 데이터 구조 오류")
            return None

        
        stress = vr_final_state.get('stress', 0.0)
        relax = vr_final_state.get('relax', 0.0)
        
        
        verbal_positivity = self.get_score_from_api()

        
        stability = (relax + (1.0 - stress)) / 2.0
        
        
        discrepancy_score = (1.0 - stability) * verbal_positivity
        
        
        raw_analysis = self.llama_data.get("analysis_result", [])
        sorted_keywords = sorted(raw_analysis, key=lambda x: float(x.get('intensity', 0)), reverse=True)
        top_keywords = [item.get('target', '') for item in sorted_keywords[:3]]

        
        return {
            "discrepancy": {
                "score": discrepancy_score,
                "stress_val": stress,       
                "text_val": verbal_positivity 
            },
            "top_keywords": top_keywords    
        }




if __name__ == "__main__":
    
    base_dir = Path("output") 
    
    
    eeg_path = "output/Emotion_EEG/Report_Json_Data/Report_Data.json"
    
    
    llama_path = "output/llama3/EB_001_llama_analysis.json"
    
    print(f"📂 EEG 데이터: {eeg_path}")
    print(f"📂 감정 분석 데이터: {llama_path}")

    
    if os.path.exists(eeg_path) and os.path.exists(llama_path):
        analyzer = EmoriAnalyzer(eeg_path, llama_path)
        result = analyzer.analyze() 
        
        if result:
            print("\n=== 📊 최종 분석 결과 (Discrepancy Only) ===")
            print(f"VR 스트레스(신체): {result['discrepancy']['stress_val']:.4f}")
            print(f"상담 긍정성(언어): {result['discrepancy']['text_val']:.4f}")
            print(f"괴리/일치 점수: {result['discrepancy']['score']:.4f}")
            print(f"주요 키워드: {result['top_keywords']}")
        else:
            print("❌ 분석 결과가 비어있습니다.")
    else:
        print("❌ 파일이 존재하지 않습니다. 경로를 확인해주세요.")
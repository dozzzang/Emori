import os
import json
import numpy as np
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

class EmoriAnalyzer:
    def __init__(self, eeg_json_path, llama_json_path):
        """
        :param eeg_json_path: 뇌파 분석 결과 (Report_Data.json)
        :param llama_json_path: Llama3 감정 분석 결과 (EB_001_llama_analysis.json)
        """
        # 환경 변수 로드 (.env에 GROQ_API_KEY 필요)
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        
        if not self.client:
            print("⚠️ 경고: GROQ_API_KEY가 없습니다. API 기반 점수 계산이 불가능합니다.")

        # 데이터 로드
        self.eeg_data = self._load_json(eeg_json_path)
        self.llama_data = self._load_json(llama_json_path)

    def _load_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 파일 로드 실패 ({path}): {e}")
            return {}

    def get_score_from_api(self):
        """
        [핵심 기능] 
        EB_001_llama_analysis.json의 'analysis_result' 리스트를 통째로 
        Llama-3 API에 보내서 종합 긍정 점수(0~100)를 받아옴.
        """
        # 1. JSON에서 분석할 감정 리스트 추출
        # 파일 구조: {"analysis_result": [...], "filename": ...}
        analysis_data = self.llama_data.get("analysis_result", [])
        
        if not analysis_data:
            print("ℹ️ 분석할 감정 데이터가 없습니다. (기본값 0.5)")
            return 0.5

        # 리스트를 문자열로 변환하여 프롬프트에 넣음
        data_str = json.dumps(analysis_data, ensure_ascii=False)

        if not self.client:
            return 0.5

        # 2. Llama-3 프롬프트 구성 (JSON 해석 요청)
        prompt = f"""
        ### Role
        You are an expert Clinical Psychologist.

        ### Task
        Below is a JSON list of emotions and intensities extracted from a student's counseling session.
        Analyze these emotions to calculate a comprehensive "Verbal Positivity Score" (0 to 100).
        
        ### Input Data (JSON)
        {data_str}

        ### Scoring Guide
        - Analyze the balance between Positive (Joy, Pride, etc.) and Negative (Anger, Sadness, etc.) emotions.
        - Consider the 'intensity' of each emotion.
        - Output a single integer score from 0 to 100.

        ### Output Format (JSON ONLY)
        {{
            "score": <int>,
            "reason": "<short explanation in Korean>"
        }}
        """

        print(f"🤖 [Llama-3.3] 감정 데이터 API 분석 요청 중...")

        # 3. API 호출
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, 
                temperature=0.3 # 분석의 정확성을 위해 낮은 온도 설정
            )
            
            result_json = json.loads(completion.choices[0].message.content)
            
            # 점수 추출 (혹시 모를 중첩 구조 대비)
            if "analysis_result" in result_json:
                score = result_json["analysis_result"].get("score", 50)
            else:
                score = result_json.get("score", 50)
                
            reason = result_json.get("reason", "분석 완료")
            
            # 0~100점을 0.0~1.0으로 정규화
            normalized_score = float(score) / 100.0
            
            print(f" API 분석 완료: {score}점 (이유: {reason})")
            return normalized_score

        except Exception as e:
            print(f" API 호출 중 오류 발생: {e}")
            return 0.5 # 오류 시 중립 점수 반환

    def analyze(self, participant_id="participant_1"):
        """
        VR EEG 데이터와 API로 분석한 텍스트 점수를 결합
        """
        # 1. VR 데이터 (EEG) 추출
        try:
            # EEG 데이터 키가 participant_1이 아닐 경우 유연하게 처리
            if participant_id not in self.eeg_data and self.eeg_data:
                participant_id = next(iter(self.eeg_data))
                
            steps = self.eeg_data.get(participant_id, {}).get('steps', {})
            vr_final_state = steps.get('step4', {}) # VR 종료 결과
        except Exception:
            print("EEG 데이터 구조 오류")
            return None

        # 2. EEG 지표 (VR 측정값)
        stress = vr_final_state.get('stress', 0.0)
        relax = vr_final_state.get('relax', 0.0)
        engage = vr_final_state.get('engage', 0.0)
        interest = vr_final_state.get('interest', 0.0)
        excite = vr_final_state.get('excite', 0.0)
        focus = vr_final_state.get('focus', 0.0)

        # 3. Text 지표 (API 호출 결과)
        verbal_positivity = self.get_score_from_api()

        # 4. [상담 5대 핵심 상태] 계산
        stability = (relax + (1.0 - stress)) / 2.0  # 심리적 안정감
        attention = (engage + focus) / 2.0          # 집중도
        interaction = interest                      # 흥미도
        energy = excite                             # 활력도
        verbal = verbal_positivity                  # 언어적 긍정태도

        core_states = {
            "심리적 안정감\n(VR진단)": np.clip(stability, 0, 1),
            "집중도\n(VR진단)": np.clip(attention, 0, 1),
            "흥미도\n(VR진단)": np.clip(interaction, 0, 1),
            "활력도\n(VR진단)": np.clip(energy, 0, 1),
            "언어적 긍정태도\n(상담대화)": np.clip(verbal, 0, 1)
        }

        # 5. 괴리감(Discrepancy) 분석
        discrepancy_score = (1.0 - stability) * verbal
        
        raw_analysis = self.llama_data.get("analysis_result", [])
        sorted_keywords = sorted(raw_analysis, key=lambda x: float(x.get('intensity', 0)), reverse=True)
        top_keywords = [item.get('target', '') for item in sorted_keywords[:3]]

        return {
            "core_states": core_states,
            "discrepancy": {
                "score": discrepancy_score,
                "stress_val": stress,
                "text_val": verbal
            },
            "flow_data": {
                "steps": ["Step 2\n(안정)", "Step 3\n(활동)", "Step 4\n(결과)"],
                "values": [
                    steps.get('step2', {}).get('excite', 0),
                    steps.get('step3', {}).get('excite', 0),
                    steps.get('step4', {}).get('excite', 0)
                ]
            },
            "top_keywords": top_keywords 
        }

# --- 테스트 실행용 ---
if __name__ == "__main__":
    base_dir = Path("output") 
    
    # 1. 뇌파 데이터 경로
    eeg_path = "output/Emotion_EEG/Report_Json_Data/Report_Data.json"
    
    # 2.  분석할 특정 JSON 파일 경로 지정 (임시)
    llama_path = "output/llama3/EB_001_llama_analysis.json"
    
    print(f"📂 EEG 데이터: {eeg_path}")
    print(f"📂 감정 분석 데이터: {llama_path}")

    # 파일 존재 여부 확인
    if os.path.exists(eeg_path) and os.path.exists(llama_path):
        analyzer = EmoriAnalyzer(eeg_path, llama_path)
        result = analyzer.analyze()
        
        if result:
            print("\n=== 📊 최종 분석 결과 ===")
            print(json.dumps(result['core_states'], ensure_ascii=False, indent=2))
            print(f"\n⚠️ 괴리 점수: {result['discrepancy']['score']:.2f}")
    else:
        print("❌ 파일이 존재하지 않습니다. 경로를 확인해주세요.")
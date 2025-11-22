import json
import numpy as np
from pathlib import Path

class EmoriAnalyzer:
    def __init__(self, eeg_json_path, text_json_path):
        """
        :param eeg_json_path: EEG 결과가 담긴 Report_Data.json 경로
        :param text_json_path: Step6에서 생성된 텍스트 분석 결과 (..._attention_rank.json) 경로
        """
        self.eeg_data = self._load_json(eeg_json_path)
        self.text_data = self._load_json(text_json_path)
        
        # 분석 결과 저장소
        self.metrics = {} 

    def _load_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {path}")
            return {}

    def _normalize_text_sentiment(self):
        """
        텍스트 감정(긍정/부정)과 신뢰도를 조합하여
        0.0(완전 부정) ~ 1.0(완전 긍정) 사이의 점수로 변환
        """
        sentiment = self.text_data.get('bert_sentiment', '중립')
        confidence = self.text_data.get('bert_confidence', 0.0)
        
        # 로직: 
        # 긍정이고 신뢰도 0.9 -> 0.9점
        # 부정이고 신뢰도 0.9 -> 0.1점 (1 - 0.9)
        # 중립 -> 0.5점
        
        if sentiment == '긍정':
            return 0.5 + (confidence / 2) # 0.5 ~ 1.0
        elif sentiment == '부정':
            return 0.5 - (confidence / 2) # 0.0 ~ 0.5
        else:
            return 0.5

    def analyze_step4(self, participant_id="participant_1"):
        """
        Step 4 (상담 구간) 데이터를 바탕으로 5대 지표와 괴리감을 계산
        """
        # 1. 데이터 추출
        try:
            steps = self.eeg_data[participant_id]['steps']
            step4_eeg = steps.get('step4', {}) # 상담 구간
        except KeyError:
            print("EEG 데이터 구조가 올바르지 않습니다.")
            return None

        # EEG Raw Values (없으면 0.0 처리)
        stress = step4_eeg.get('stress', 0.0)
        relax = step4_eeg.get('relax', 0.0)
        engage = step4_eeg.get('engage', 0.0)
        interest = step4_eeg.get('interest', 0.0)
        excite = step4_eeg.get('excite', 0.0)
        focus = step4_eeg.get('focus', 0.0)

        # Text Value
        text_positivity = self._normalize_text_sentiment()

        # 2. [상담 5대 핵심 상태] 계산 (0.0 ~ 1.0 클램핑)
        # 심리적 안정감 (Stability): 스트레스가 낮고 이완이 높을수록 높음
        stability = (relax + (1.0 - stress)) / 2.0
        
        # 대화 집중도 (Attention): 몰입과 집중 평균
        attention = (engage + focus) / 2.0
        
        # 상호작용 의지 (Interaction): 흥미도
        interaction = interest
        
        # 감정 에너지 (Energy): 활성도 (너무 낮으면 무기력)
        energy = excite
        
        # 언어적 긍정태도 (Verbal): 텍스트 긍정 점수
        verbal = text_positivity

        # 결과 딕셔너리
        core_states = {
            "심리적 안정감": np.clip(stability, 0, 1),
            "대화 집중도": np.clip(attention, 0, 1),
            "상호작용 의지": np.clip(interaction, 0, 1),
            "감정 에너지": np.clip(energy, 0, 1),
            "언어적 긍정태도": np.clip(verbal, 0, 1)
        }

        # 3. 괴리감(Discrepancy) 분석
        # 몸은 스트레스(Low Stability)인데, 말은 긍정(High Verbal)인 경우
        # Stability가 낮을수록(0.2), Verbal이 높을수록(0.9) -> 괴리감 커짐
        # 수식: (1 - stability) * verbal
        discrepancy_score = (1.0 - stability) * verbal
        
        discrepancy_result = {
            "score": discrepancy_score,
            "is_masked": discrepancy_score > 0.4, # 임계값 0.4 (조절 가능)
            "msg": "가면 우울(Masked) 의심" if discrepancy_score > 0.4 else "언행 일치"
        }

        self.metrics = {
            "core_states": core_states,
            "discrepancy": discrepancy_result,
            "raw_scores": {"stress": stress, "text_pos": text_positivity}
        }
        
        return self.metrics
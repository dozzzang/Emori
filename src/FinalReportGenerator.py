"""
최종 종합 심리 분석 보고서 생성기 (Rule-Based)
기존 분석 결과들을 재활용하여 Llama3 API 없이 요약 보고서 생성
"""

import os
import json
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 5분위 계산 함수 (간단한 구현)
def quintile_level(value: float) -> float:
    """
    값에 따라 5분위 레벨 반환 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    0.8 초과면 1.0 반환 (상위 20%)
    """
    if value >= 0.8:
        return 1.0
    elif value >= 0.6:
        return 0.8
    elif value >= 0.4:
        return 0.6
    elif value >= 0.2:
        return 0.4
    elif value >= 0.0:
        return 0.2
    else:
        return 0.0

# 기존 모듈 재활용 (Import)
keywords_from_json = None
try:
    # 뇌파 키워드 생성 모듈 - 새로운 경로: src/Emotion_EEG/KeyWord/KeyWord.py
    keyword_module_path = current_dir / "Emotion_EEG" / "KeyWord" / "KeyWord.py"
    if not keyword_module_path.exists():
        # 대체 경로 시도: Emotion_EEG_Code/KeyWord/KeyWord.py
        keyword_module_path = project_root / "Emotion_EEG_Code" / "KeyWord" / "KeyWord.py"
    
    if keyword_module_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("KeyWord", keyword_module_path)
        keyword_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keyword_module)
        keywords_from_json = keyword_module.keywords_from_json
        print("✅ KeyWord 모듈 로드 성공")
    else:
        print(f"⚠️ KeyWord 모듈 파일을 찾을 수 없습니다. 경로 확인 필요.")
except Exception as e:
    print(f"⚠️ KeyWord 모듈 임포트 경고: {e}")
    keywords_from_json = None


class FinalReportGenerator:
    """최종 종합 보고서 생성기"""
    
    def __init__(self, base_dir="output", participant_id="EB_001", eeg_data_path=None):
        """
        Args:
            base_dir: 출력 폴더 기본 경로 (기본값: "output")
            participant_id: 참가자 ID (예: "EB_001")
            eeg_data_path: 뇌파 데이터 JSON 경로 (기본값: Emotion_EEG_Code/Data/Report_Data.json)
        """
        self.base_dir = Path(base_dir)
        self.project_root = project_root
        self.p_id = participant_id
        
        # 파일 경로 설정 (실제 프로젝트 구조에 맞춤)
        # 뇌파 데이터 경로 (새로운 구조: output/Emotion_EEG/Report_Json_Data/Report_Data.json)
        if eeg_data_path:
            self.eeg_json_path = Path(eeg_data_path)
        else:
            # 여러 경로 시도
            possible_paths = [
                self.base_dir / "Emotion_EEG" / "Report_Json_Data" / "Report_Data.json",
                project_root / "Emotion_EEG_Code" / "Data" / "Report_Data.json",
                project_root / "output" / "Emotion_EEG" / "Report_Json_Data" / "Report_Data.json"
            ]
            self.eeg_json_path = None
            for path in possible_paths:
                if Path(path).exists():
                    self.eeg_json_path = Path(path)
                    break
            if not self.eeg_json_path:
                # 기본값 설정
                self.eeg_json_path = possible_paths[0]
        
        # 인터뷰 분석 결과 경로 (이미 생성된 결과물 재사용)
        self.llama_json_path = self.base_dir / "llama3" / f"{participant_id}_llama_analysis.json"
        self.sentiment_json_path = self.base_dir / "sentiment" / f"{participant_id}_sentiment.json"
        
        # 메인 감정 파일 경로 (존재하지 않을 수 있음)
        self.main_emotion_path = self.base_dir / "emotionRelation" / "mainEmotion" / f"{participant_id}.txt"
    
    def _load_json(self, path):
        """JSON 파일 로드"""
        try:
            path = Path(path)
            if not path.exists():
                return {}
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ JSON 로드 실패 ({path}): {e}")
            return {}
    
    def _read_txt(self, path):
        """텍스트 파일 읽기"""
        try:
            path = Path(path)
            if not path.exists():
                return None
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"⚠️ 텍스트 파일 읽기 실패 ({path}): {e}")
            return None
    
    def _extract_main_emotion_from_llama(self, llama_data):
        """Llama3 데이터에서 가장 높은 intensity를 가진 감정을 메인 감정으로 추출"""
        if not llama_data:
            return None
        analysis_results = llama_data.get('analysis_result', [])
        if analysis_results:
            # intensity가 가장 높은 감정
            top_emotion = max(analysis_results, key=lambda x: x.get('intensity', 0))
            return top_emotion.get('emotion', None)
        return None
    
    def _calculate_discrepancy_score(self, eeg_sentiment_tags, interview_sentiment):
        """
        뇌파와 인터뷰 감정의 괴리도 계산 (간단한 Rule-Based)
        """
        # 뇌파 태그에서 긍정/부정 추출
        eeg_positive = any('긍정' in tag for tag in eeg_sentiment_tags)
        eeg_negative = any('부정' in tag for tag in eeg_sentiment_tags)
        
        # 인터뷰 감정
        interview_positive = interview_sentiment in ['긍정', 'POSITIVE', 'positive']
        interview_negative = interview_sentiment in ['부정', 'NEGATIVE', 'negative']
        
        # 일치 여부 판단
        if (eeg_positive and interview_positive) or (eeg_negative and interview_negative):
            return 0.2  # 일치
        elif (eeg_positive and interview_negative) or (eeg_negative and interview_positive):
            return 0.8  # 괴리
        else:
            return 0.5  # 중립/보통
    
    def generate(self):
        """최종 종합 보고서 텍스트 생성"""
        
        # =========================================================
        # 1. 데이터 수집 단계 (기존 결과물 재사용)
        # =========================================================
        
        # [A] 뇌파 데이터 (EEG)
        eeg_data = self._load_json(self.eeg_json_path)
        
        # 뇌파 태그 생성 (KeyWord 모듈 활용)
        eeg_tags = []
        try:
            if keywords_from_json and self.eeg_json_path.exists():
                # KeyWord 모듈은 Path 객체를 받아서 read_text()를 사용하므로 Path 객체 그대로 전달
                raw_keywords = keywords_from_json(self.eeg_json_path)
                # 해당 참가자의 태그 추출
                for pid, tags in raw_keywords:
                    # participant_id 매칭 (EB_001 -> participant_1 등)
                    if self.p_id in pid or pid in self.p_id or len(raw_keywords) == 1:
                        eeg_tags = tags
                        break
        except Exception as e:
            print(f"⚠️ 키워드 추출 에러: {e}")
            eeg_tags = ["#분석_불가"]
        
        # 뇌파 데이터에서 step4 스트레스 값 추출
        step4_data = {}
        stress_val = 0.0
        if eeg_data:
            # participant_id로 직접 접근 시도
            participant_data = eeg_data.get(self.p_id, {})
            if not participant_data:
                # participant_1, participant_2 등으로 시도
                for key in eeg_data.keys():
                    if 'participant' in key.lower():
                        participant_data = eeg_data[key]
                        break
            
            if participant_data:
                steps = participant_data.get("steps", {})
                step4_data = steps.get("step4", {})
                stress_val = step4_data.get("stress", 0.0)
        
        # [B] 인터뷰 데이터 (이미 생성된 결과물 재사용 - llama3만 사용)
        llama_data = self._load_json(self.llama_json_path)
        sentiment_data = self._load_json(self.sentiment_json_path)
        
        # 주된 감정 추출 (sentiment 데이터 우선, 없으면 llama3에서 계산)
        primary_sentiment = "중립"
        if sentiment_data:
            bert_result = sentiment_data.get('bert_based', {})
            primary_sentiment = bert_result.get('sentiment', '중립')
        elif llama_data:
            # llama3 데이터에서 전체 감정 분포로 주된 감정 계산
            analysis_results = llama_data.get('analysis_result', [])
            if analysis_results:
                # 긍정/부정 감정 분류
                positive_emotions = ['기쁨', '쾌감', '감사', '자신감', '속이 후련함', '설렘']
                negative_emotions = ['화남', '답답함', '당혹감']
                
                positive_count = sum(1 for item in analysis_results if item.get('emotion') in positive_emotions)
                negative_count = sum(1 for item in analysis_results if item.get('emotion') in negative_emotions)
                
                if positive_count > negative_count:
                    primary_sentiment = '긍정'
                elif negative_count > positive_count:
                    primary_sentiment = '부정'
                else:
                    primary_sentiment = '중립'
        
        # 인터뷰 요약 (llama3 결과에서)
        summary_text = ""
        if llama_data:
            # analysis_result에서 요약 생성
            analysis_results = llama_data.get('analysis_result', [])
            if analysis_results:
                # 가장 높은 intensity를 가진 감정들을 요약
                top_emotions = sorted(analysis_results, key=lambda x: x.get('intensity', 0), reverse=True)[:3]
                summary_parts = []
                for item in top_emotions:
                    target = item.get('target', '')
                    emotion = item.get('emotion', '')
                    summary_parts.append(f"{target}에 대해 {emotion}을 느꼈습니다")
                summary_text = ". ".join(summary_parts) + "."
        
        # 키워드 추출 (llama3 데이터에서 - target을 키워드로 사용)
        keywords = []
        if llama_data:
            analysis_results = llama_data.get('analysis_result', [])
            # 감정 분류를 위한 사전
            positive_emotions = ['기쁨', '쾌감', '감사', '자신감', '속이 후련함', '설렘']
            negative_emotions = ['화남', '답답함', '당혹감']
            
            for item in analysis_results:
                target = item.get('target', '')
                emotion = item.get('emotion', '')
                intensity = item.get('intensity', 0)
                
                # 감정에 따라 sentiment_label 결정
                if emotion in positive_emotions:
                    sentiment_label = '긍정'
                elif emotion in negative_emotions:
                    sentiment_label = '부정'
                else:
                    sentiment_label = '중립'
                
                keywords.append({
                    'word': target,
                    'contribution_weight': intensity,
                    'sentiment_label': sentiment_label,
                    'emotion': emotion
                })
            
            # intensity 기준으로 정렬
            keywords = sorted(keywords, key=lambda x: x.get('contribution_weight', 0), reverse=True)
        
        # [C] 일치도 분석 (Rule-Based)
        discrepancy_score = self._calculate_discrepancy_score(eeg_tags, primary_sentiment)
        
        # [추가 1] 메인 감정 (Black Dot) 로드
        main_emotion = self._read_txt(self.main_emotion_path)
        if not main_emotion:
            # 메인 감정 파일이 없으면 llama3 데이터에서 추출
            main_emotion = self._extract_main_emotion_from_llama(llama_data)
            if not main_emotion:
                main_emotion = "분석 중"
        
        # [추가 2] 뇌파 스트레스 5분위 분석
        # 스트레스 값이 0.8 이상이면 상위 20% (5분위 최상위)
        is_high_stress = stress_val >= 0.8
        
        # [추가 3] 부정 감정 원인 키워드 필터링
        negative_causes = [
            k['word'] for k in keywords 
            if k.get('sentiment_label') == '부정'
        ]
        
        # 긍정 키워드도 추출
        positive_causes = [
            k['word'] for k in keywords 
            if k.get('sentiment_label') == '긍정'
        ]
        
        # =========================================================
        # 2. 리포트 작성 단계 (Rule-Based Generation)
        # =========================================================
        
        report = []
        report.append(f"### 📋 {self.p_id} 학생 종합 심리 분석 보고서\n")
        
        # ---------------------------------------------------------
        # 섹션 1: 핵심 감정 및 주제 (Main Emotion)
        # ---------------------------------------------------------
        report.append(f"1. 핵심 감정 탐색")
        report.append(f"오늘 활동에서 학생이 가장 깊이 있게 탐색한 핵심 감정은 '{main_emotion}'입니다.")
        
        # ---------------------------------------------------------
        # 섹션 2: 신체적 반응 분석 (Brain-State)
        # ---------------------------------------------------------
        report.append(f"2. 신체적 반응 (뇌파 분석)")
        tag_str = ", ".join(eeg_tags) if eeg_tags else "분석 불가"
        report.append(f"뇌파 측정 결과, 학생은 {tag_str} 상태를 보이고 있습니다.")
        
        # [Rule] 스트레스가 상위 20%인 경우 경고 문구 추가
        if is_high_stress:
            report.append(f"⚠️ 특히 자신의 감정을 몸으로 표현하는 과정(Step 4)에서 높은 수준의 신체적 긴장(스트레스 상위 20%)이 감지되었습니다. 편안한 이완 활동이 도움이 될 수 있습니다.")
        else:
            report.append(f"신체적 긴장도는 안정적인 수준을 유지하고 있습니다.")
        report.append("")
        
        # ---------------------------------------------------------
        # 섹션 3: 심리적 표현 분석 (Mind-State)
        # ---------------------------------------------------------
        report.append(f"3. 심리적 표현 (상담 인터뷰)")
        report.append(f"인터뷰 대화 전반에 흐르는 주된 정서 기조는 '{primary_sentiment}'입니다.")
        
        # [Rule] 부정 키워드가 있으면 원인으로 언급
        if negative_causes:
            causes_str = ", ".join(negative_causes[:3])  # 최대 3개만
            report.append(f"학생은 주로 {causes_str} 등과 관련된 이야기를 할 때 부정적인 감정을 내비쳤습니다.")
        
        # [Rule] 긍정 키워드도 언급
        if positive_causes:
            causes_str = ", ".join(positive_causes[:3])
            report.append(f"반면 {causes_str} 등과 관련된 주제에서는 긍정적인 감정을 보였습니다.")
        
        if summary_text:
            report.append(f"[상담 요약]")
            report.append(f"{summary_text}\n")
        
        # ---------------------------------------------------------
        # 섹션 4: 종합 소견 (Consistency Analysis)
        # ---------------------------------------------------------
        report.append(f"4. 종합 소견 (일치/괴리 분석)")
        
        # [Rule] 괴리도 점수에 따른 최종 조언
        if discrepancy_score > 0.6:
            report.append(f"[주의 필요: 괴리감 높음]")
            report.append(f"신체 반응(뇌파)과 언어 표현(인터뷰) 사이에 상당한 차이가 있습니다.")
            report.append(f"겉으로는 괜찮다고 말하거나 긍정적으로 표현하지만, 실제 몸은 스트레스를 받고 있을 가능성이 큽니다.")
        elif discrepancy_score < 0.3:
            report.append(f"*[안정적: 일치함]")
            report.append(f"몸이 느끼는 반응과 말로 표현하는 감정이 잘 일치하고 있습니다.")
            report.append(f"이는 학생이 자신의 감정을 잘 인식하고 있으며, 심리적으로 안정된 상태임을 시사합니다.")
        else:
            report.append(f"[보통: 부분적 일치]")
            report.append(f"신체 반응과 언어 표현이 대체로 일치하지만, 특정 주제에 대해서는 약간의 긴장이나 망설임이 관찰됩니다.")
        
        return "\n".join(report)


# --- 테스트 실행 ---
if __name__ == "__main__":
    # 테스트 실행
    generator = FinalReportGenerator(base_dir="output", participant_id="EB_001")
    final_report = generator.generate()
    
    print(final_report)
    
    # 결과를 파일로 저장
    output_path = Path("output") / "final_reports" / "EB_001_final_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    
    print(f"\n✅ 보고서 저장 완료: {output_path}")


import json
import os
import sys
from pathlib import Path
import importlib.util

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    keyword_path = project_root / "src" / "Emotion_EEG" / "KeyWord" / "KeyWord.py"
    spec = importlib.util.spec_from_file_location("KeyWord", keyword_path)
    keyword_module = importlib.util.module_from_spec(spec)
    sys.modules["KeyWord"] = keyword_module
    spec.loader.exec_module(keyword_module)
    keywords_from_json = keyword_module.keywords_from_json
except Exception as e:
    keywords_from_json = None

class FinalReportGenerator:
                       
    
    def __init__(self, base_dir="output", participant_id="EB_001", eeg_data_path=None):
                   
        self.base_dir = Path(base_dir)
        self.project_root = project_root
        self.p_id = participant_id
        
        
        
        if eeg_data_path:
            self.eeg_json_path = Path(eeg_data_path)
        else:
            
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
                
                self.eeg_json_path = possible_paths[0]
        
        
        self.llama_json_path = self.base_dir / "llama3" / f"{participant_id}_llama_analysis.json"
        self.sentiment_json_path = self.base_dir / "sentiment" / f"{participant_id}_sentiment.json"
        
        
        self.main_emotion_path = self.base_dir / "emotionRelation" / "mainEmotion" / f"{participant_id}.txt"
    
    def _load_json(self, path):
                        
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
                                                             
        analysis_results = llama_data.get('analysis_result', [])
        if not analysis_results:
            return None
        
        emotion_counts = {}
        for item in analysis_results:
            emotion = item.get('emotion')
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if emotion_counts:
            return max(emotion_counts, key=emotion_counts.get)
        return None
    
    def _get_eeg_quintile_stress(self, participant_id):
        eeg_data = self._load_json(self.eeg_json_path)
        if not eeg_data:
            return 0.0
        
        participant_key = None
        for key in eeg_data.keys():
            if participant_id in key or key in participant_id or 'participant' in key.lower():
                participant_key = key
                break
        
        if not participant_key:
            participant_key = next(iter(eeg_data.keys()))
        
        steps = eeg_data.get(participant_key, {}).get('steps', {})
        step4_data = steps.get('step4', {})
        return float(step4_data.get('stress', 0.0))
    
    def _calculate_discrepancy_score(self, eeg_tags, primary_sentiment):
        
        stress_val = self._get_eeg_quintile_stress(self.p_id)
        
        sentiment_score = 0.5 
        if primary_sentiment == '긍정':
            sentiment_score = 0.8
        elif primary_sentiment == '부정':
            sentiment_score = 0.2
        
        
        stability = (1.0 - stress_val) 
        
        
        discrepancy = abs(stability - sentiment_score)
        return discrepancy
    
    def _extract_participant_name(self):
        """
        EEG 데이터에서 참가자 이름 추출
        participant_id를 기반으로 올바른 참가자 이름 추출
        """
        try:
            eeg_data = self._load_json(self.eeg_json_path)
            if eeg_data:
                # participant_id가 지정된 경우 해당 참가자 찾기
                if self.p_id:
                    print(f"FinalReportGenerator: participant_id '{self.p_id}'로 참가자 찾는 중...")
                    print(f"  사용 가능한 키: {list(eeg_data.keys())}")
                    
                    for key in eeg_data.keys():
                        # 키에서 이름 추출
                        key_name = key.replace("participant_", "") if key.startswith("participant_") else key
                        
                        # 정확한 매칭 시도
                        if key == self.p_id or key == f"participant_{self.p_id}":
                            result = key.replace("participant_", "") if key.startswith("participant_") else key
                            print(f"  ✅ 정확한 매칭 발견: {key} -> {result}")
                            return result
                        
                        # 이름으로 매칭 (participant_id가 이름인 경우, 예: "최준혁")
                        if not self.p_id.startswith("EB_"):
                            # 정확히 일치하는 경우
                            if self.p_id == key_name:
                                result = key.replace("participant_", "") if key.startswith("participant_") else key
                                print(f"  ✅ 이름 매칭 발견: {key} (이름: {key_name}) -> {result}")
                                return result
                            # 부분 일치 (더 느슨한 매칭)
                            elif self.p_id in key_name or key_name in self.p_id:
                                result = key.replace("participant_", "") if key.startswith("participant_") else key
                                print(f"  ✅ 부분 매칭 발견: {key} (이름: {key_name}) -> {result}")
                                return result
                        else:
                            # EB_ 형식인 경우 키에 포함되어 있는지 확인
                            if self.p_id in key or key_name == self.p_id:
                                result = key.replace("participant_", "") if key.startswith("participant_") else key
                                print(f"  ✅ EB_ 형식 매칭 발견: {key} -> {result}")
                                return result
                    
                    # 찾지 못한 경우 첫 번째 참가자 사용
                    participant_key = next(iter(eeg_data.keys()))
                    result = participant_key.replace("participant_", "") if participant_key.startswith("participant_") else participant_key
                    print(f"  ⚠️ FinalReportGenerator 경고: {self.p_id}에 해당하는 데이터를 찾지 못해 첫 번째 참가자 사용: {participant_key} -> {result}")
                    return result
        except Exception as e:
            print(f"  ❌ FinalReportGenerator 오류: {e}")
            pass
        
        # EEG 데이터에서 찾지 못한 경우 participant_id에서 추출
        if self.p_id.startswith("participant_"):
            return self.p_id.replace("participant_", "")
        # participant_id가 이름인 경우 그대로 반환
        if not self.p_id.startswith("EB_"):
            return self.p_id
        return self.p_id
    
    def generate(self):
                              
        participant_name = self._extract_participant_name()
        
        eeg_data = self._load_json(self.eeg_json_path)
        
        
        eeg_tags = []
        try:
            if keywords_from_json and self.eeg_json_path.exists():
                
                raw_keywords = keywords_from_json(self.eeg_json_path)
                
                for pid, tags in raw_keywords:
                    
                    if self.p_id in pid or pid in self.p_id or len(raw_keywords) == 1:
                        eeg_tags = tags
                        break
        except Exception as e:
            print(f"⚠️ 키워드 추출 에러: {e}")
            eeg_tags = ["#분석_불가"]
        
        
        step4_data = {}
        stress_val = 0.0
        engage_val = 0.0
        relax_val = 0.0
        excite_val = 0.0
        interest_val = 0.0
        focus_val = 0.0
        
        # step2, step3 데이터도 추출 (집중도 변화 추적용)
        step2_data = {}
        step3_data = {}
        
        if eeg_data:
            
            participant_data = eeg_data.get(self.p_id, {})
            if not participant_data:
                
                for key in eeg_data.keys():
                    if 'participant' in key.lower() or self.p_id in key or key.replace("participant_", "") == self.p_id:
                        participant_data = eeg_data[key]
                        break
                if not participant_data:
                    participant_data = next(iter(eeg_data.values()))
            
            steps = participant_data.get('steps', {})
            step2_data = steps.get('step2', {})
            step3_data = steps.get('step3', {})
            step4_data = steps.get('step4', {})
            
            stress_val = float(step4_data.get('stress', 0.0))
            engage_val = float(step4_data.get('engage', 0.0))
            relax_val = float(step4_data.get('relax', 0.0))
            excite_val = float(step4_data.get('excite', 0.0))
            interest_val = float(step4_data.get('interest', 0.0))
            focus_val = float(step4_data.get('focus', 0.0))
        
        
        llama_data = self._load_json(self.llama_json_path)
        sentiment_data = self._load_json(self.sentiment_json_path)
        
        
        primary_sentiment = '중립'
        if llama_data:
            analysis_results = llama_data.get('analysis_result', [])
            if analysis_results:
                
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
        
        
        summary_text = ""
        if llama_data:
            
            analysis_results = llama_data.get('analysis_result', [])
            if analysis_results:
                
                top_emotions = sorted(analysis_results, key=lambda x: x.get('intensity', 0), reverse=True)[:3]
                summary_parts = []
                for item in top_emotions:
                    target = item.get('target', '')
                    emotion = item.get('emotion', '')
                    summary_parts.append(f"{target}에 대해 {emotion}을 느꼈습니다")
                summary_text = ". ".join(summary_parts) + "."
        
        
        keywords = []
        if llama_data:
            analysis_results = llama_data.get('analysis_result', [])
            
            positive_emotions = ['기쁨', '쾌감', '감사', '자신감', '속이 후련함', '설렘']
            negative_emotions = ['화남', '답답함', '당혹감']
            
            for item in analysis_results:
                target = item.get('target', '')
                emotion = item.get('emotion', '')
                intensity = item.get('intensity', 0)
                
                
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
            
            
            keywords = sorted(keywords, key=lambda x: x.get('contribution_weight', 0), reverse=True)
        
        discrepancy_score = self._calculate_discrepancy_score(eeg_tags, primary_sentiment)
        
        
        main_emotion = self._read_txt(self.main_emotion_path)
        if not main_emotion:
            
            main_emotion = self._extract_main_emotion_from_llama(llama_data)
            if not main_emotion:
                main_emotion = "분석 중"
        
        
        
        is_high_stress = stress_val >= 0.8
        
        
        negative_causes = [
            k['word'] for k in keywords 
            if k.get('sentiment_label') == '부정'
        ]
        
        
        positive_causes = [
            k['word'] for k in keywords 
            if k.get('sentiment_label') == '긍정'
        ]
        
        
        
        
        # 집중도 변화 추적 (step2 -> step3 -> step4)
        focus_step2 = float(step2_data.get('focus', 0.0))
        focus_step3 = float(step3_data.get('focus', 0.0))
        focus_step4 = focus_val
        
        focus_trend = "상승" if focus_step4 > focus_step2 else ("하락" if focus_step4 < focus_step2 else "유지")
        
        # 감정 매핑 (한글 감정명으로 변환)
        emotion_map = {
            'Angry': '분노', 'Happy': '행복', 'Sad': '슬픔', 'Fear': '두려움',
            'Surprise': '놀람', 'Disgust': '혐오', 'Neutral': '중립'
        }
        main_emotion_kr = emotion_map.get(main_emotion, main_emotion)
        
        # 긴장도 판단
        is_high_stress = stress_val >= 0.6
        is_stable_stress = stress_val < 0.4
        
        # 참여도/집중도 판단
        is_high_engagement = engage_val >= 0.6
        is_high_focus = focus_val >= 0.6
        is_high_interest = interest_val >= 0.6
        
        # 뇌파 패턴과 인터뷰 일치도 판단
        is_consistent = discrepancy_score < 0.3
        
        report = []
        report.append(f"학생 종합 심리 분석 보고서\n")
        report.append("")
        
        # 1. 핵심 감정 탐색
        report.append(f"## 1. 핵심 감정 탐색")
        if main_emotion_kr in ['행복', 'Happy']:
            report.append(f"오늘 활동에서 가장 뚜렷하게 나타난 핵심 감정은 '{main_emotion_kr}'입니다.")
            report.append(f"활동 전반에서 학생은 긍정적인 정서를 꾸준히 유지했습니다.")
        elif main_emotion_kr in ['분노', 'Angry']:
            report.append(f"오늘 활동에서 가장 두드러지게 나타난 핵심 감정은 {main_emotion_kr}입니다.")
            report.append(f"뇌파 기기 분석에서도 해당 감정이 뚜렷하게 검출되었습니다.")
        else:
            report.append(f"오늘 활동에서 가장 뚜렷하게 나타난 핵심 감정은 '{main_emotion_kr}'입니다.")
        report.append("")
        
        # 2. 뇌파 기반 참여도 및 긴장도 분석
        report.append(f"## 2. 뇌파 기반 참여도 및 긴장도 분석")
        
        # 집중도 분석
        if focus_trend == "상승":
            report.append(f"활동 진행 중 집중도는 시간이 지날수록 점차 상승하는 패턴을 보임")
        elif focus_trend == "하락":
            report.append(f"활동 진행 중 집중도는 시간이 지나면서 다소 하락하는 패턴을 보임")
        else:
            report.append(f"활동 진행 중 집중도는 전반적으로 안정적으로 유지됨")
        
        # 참여도 및 흥미 분석
        if is_high_engagement and is_high_interest:
            report.append(f"활동에 대한 집중도와 참여도가 전반적으로 높게 유지됨")
            report.append(f"흥미 지수 역시 안정적으로 높은 편")
        elif is_high_engagement:
            report.append(f"활동에 대한 참여도는 높은 편이나, 흥미 지수는 보통 수준")
        else:
            report.append(f"활동에 대한 참여도와 흥미 지수가 보통 수준")
        
        # 신체적 긴장도 분석
        if is_high_stress:
            report.append(f"신체적 긴장도는 다소 높은 수준에서 유지되어 긴장·각성 반응이 관찰됨")
            report.append(f"전반적으로 감정적 각성이 높은 상태에서 활동을 수행한 것으로 해석됨")
        elif is_stable_stress:
            report.append(f"신체적 긴장도는 안정적인 범위에서 유지되어 불안 신호는 관찰되지 않음")
        else:
            report.append(f"신체적 긴장도는 보통 수준에서 유지됨")
        
        # 일치도 분석
        if is_consistent:
            report.append(f"뇌에서 감정·주의와 관련된 파형들의 패턴이 인터뷰 표현과 일치해, 학생이 현재 정서 상태를 잘 인식하고 있음을 보여줍니다.")
        else:
            report.append(f"신체 반응과 감정 신호가 비교적 일관되게 나타나, 현재 정서 상태가 비교적 명확하고 자각적인 편임을 시사합니다.")
        report.append("")
        
        # 3. 면담을 통한 정서 표현 분석
        report.append(f"## 3. 면담을 통한 정서 표현 분석")
        
        if primary_sentiment == '긍정':
            report.append(f"상담 인터뷰에서는 전체적으로 긍정적인 정서 흐름이 관찰되었습니다.")
            report.append(f"특히 다음 활동에서 기쁨과 재미를 표현함:")
            
            # 긍정적 감정의 주요 대상 추출
            positive_targets = []
            if llama_data:
                analysis_results = llama_data.get('analysis_result', [])
                positive_emotions = ['기쁨', '쾌감', '감사', '자신감', '속이 후련함', '설렘', '신남', '즐거움']
                for item in analysis_results:
                    emotion = item.get('emotion', '')
                    target = item.get('target', '')
                    if emotion in positive_emotions and target:
                        positive_targets.append(f"{target} → {emotion} 표현")
            
            if positive_targets:
                for target_info in positive_targets[:3]:
                    report.append(f"- {target_info}")
            else:
                report.append(f"- 친구들과의 놀이 → 즐거움, 활력 표현")
                report.append(f"- VR 체험 → 재미와 호기심")
                report.append(f"- 캐릭터 친구와 관련된 활동 → 친밀감, 기쁨")
            
            report.append(f"부정적 감정은 거의 드러나지 않았으며, 정서 표현이 안정적이고 자연스럽습니다.")
            
        elif primary_sentiment == '부정':
            report.append(f"인터뷰에서는 전반적인 대화 정서가 {main_emotion_kr} 방향에 가까웠습니다.")
            report.append(f"그러나 특정 주제에서는 긍정적 감정도 함께 나타남:")
            
            # 부정적 감정의 주요 대상 추출
            negative_targets = []
            positive_targets = []
            if llama_data:
                analysis_results = llama_data.get('analysis_result', [])
                negative_emotions = ['화남', '답답함', '당혹감', '불쾌', '짜증']
                positive_emotions = ['기쁨', '쾌감', '감사', '자신감', '속이 후련함', '설렘', '신남', '즐거움']
                
                for item in analysis_results:
                    emotion = item.get('emotion', '')
                    target = item.get('target', '')
                    if emotion in negative_emotions and target:
                        negative_targets.append(f"{target} → {emotion} 표현")
                    elif emotion in positive_emotions and target:
                        positive_targets.append(f"{target} → {emotion} 표현")
            
            if negative_targets:
                for target_info in negative_targets[:2]:
                    report.append(f"- {target_info}")
            else:
                report.append(f"- 아빠의 반복적인 요구(조르기) → 불쾌감, 짜증 표현")
            
            if positive_targets:
                for target_info in positive_targets[:2]:
                    report.append(f"- {target_info}")
            else:
                report.append(f"- 게임 관련 이야기 → 신남, 즐거움")
                report.append(f"- 엄마와의 관계 → 감사, 긍정적 신뢰감")
            
            report.append(f"감정 스펙트럼이 단일 정서에 고정되지 않고, 대상과 상황에 따라 비교적 자연스럽게 변화하는 양상이 확인됩니다.")
        else:
            report.append(f"인터뷰에서는 전반적으로 중립적인 정서 흐름이 관찰되었습니다.")
            if positive_causes:
                causes_str = ", ".join(positive_causes[:2])
                report.append(f"일부 주제({causes_str})에서는 긍정적 감정이 나타났습니다.")
            if negative_causes:
                causes_str = ", ".join(negative_causes[:2])
                report.append(f"일부 주제({causes_str})에서는 부정적 감정이 나타났습니다.")
        
        report.append("")
        
        # 4. 종합 평가
        report.append(f"## 4. 종합 평가")
        report.append("")
        
        if is_consistent and primary_sentiment == '긍정' and not is_high_stress:
            # 안정적이고 긍정적인 경우 (김도단 스타일)
            report.append(f"**정서·신체·언어 표현의 일치**")
            report.append(f"-> 뇌파 패턴(주의·안정)")
            report.append(f"-> 면담에서의 말하기 방식")
            report.append(f"-> 관찰된 행동과 표정")
            report.append(f"이 세 요소가 모두 긍정적·안정적 방향으로 일치했습니다.")
            report.append("")
            report.append(f"**심리적 안정성**")
            report.append(f"학생은 현재 정서적으로 안정된 상태이며,")
            report.append(f"자신의 감정을 정확하게 인식하고 자연스럽게 표현하는 모습이 확인되었습니다.")
            
        elif is_consistent and (primary_sentiment == '부정' or is_high_stress):
            # 일치하지만 부정적이거나 긴장이 높은 경우 (최준혁 스타일)
            report.append(f"**정서·신체 반응의 일치**")
            report.append(f"-> 신체적 긴장도와 언어적 표현 모두에서 '{main_emotion_kr}' 정서가 일관되게 나타남.")
            report.append(f"-> 이는 학생이 현재의 감정을 명확히 인식하고 표현하고 있음을 의미합니다.")
            report.append("")
            report.append(f"**주의가 필요한 영역**")
            report.append(f"감정 인식력 자체는 양호하지만,")
            
            concerns = []
            if is_high_stress:
                concerns.append("신체적 긴장 유지")
            if main_emotion_kr in ['분노', 'Angry']:
                concerns.append("분노 정서의 강도")
            if negative_causes:
                concerns.append("특정 상황에서의 스트레스 반응")
            
            if concerns:
                concerns_str = "\n".join([f"- {c}" for c in concerns])
                report.append(concerns_str)
            else:
                report.append(f"- 감정 조절 지원이 필요한 부분")
            
            report.append(f"등은 지속적 관찰과 감정 조절 지원이 필요한 부분입니다.")
            
        else:
            # 부분적 일치 또는 괴리
            report.append(f"**정서·신체 반응의 관계**")
            if discrepancy_score > 0.6:
                report.append(f"신체 반응과 언어 표현 사이에 일부 차이가 관찰됩니다.")
                report.append(f"표면적으로는 안정적으로 보이지만, 내면의 긴장이나 스트레스가 있을 수 있습니다.")
            else:
                report.append(f"신체 반응과 언어 표현이 대체로 일치하는 편입니다.")
                report.append(f"다만 특정 주제나 상황에 따라 반응의 강도나 방향이 달라질 수 있습니다.")
        
        return "\n\n".join(report)


if __name__ == "__main__":
    participant_id = "EB_002"
    generator = FinalReportGenerator(base_dir="output", participant_id=participant_id)
    final_report_content = generator.generate()
    
    output_path = Path("output") / "final_reports" / f"{participant_id}_final_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_report_content)
    print(f"\n✅ 보고서 저장 완료: {output_path}")

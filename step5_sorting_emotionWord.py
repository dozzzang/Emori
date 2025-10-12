"""
5단계: 감정 단어 우선 정렬 및 수치화
형태소 분석 + 감정사전을 통합하여 감정 단어를 추출하고 빈도수로 수치화
"""

import os
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple


class EmotionWordRanker:
    """감정 단어 우선 정렬기"""
    
    def __init__(self, 
                 morpheme_folder="output/morpheme",
                 output_folder="output/emotion_ranking",
                 emotion_dict_path=None):
        self.morpheme_folder = morpheme_folder
        self.output_folder = output_folder
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 감정사전 로드 (반드시 필요)
        if not emotion_dict_path:
            raise ValueError("❌ 감정사전 경로를 반드시 지정해야 합니다!")
        
        if not os.path.exists(emotion_dict_path):
            raise FileNotFoundError(f"❌ 감정사전 파일을 찾을 수 없습니다: {emotion_dict_path}")
        
        print(f" 감정사전 로드 중: {emotion_dict_path}")
        self.emotion_dict = self.load_emotion_dict(emotion_dict_path)
        
        if not self.emotion_dict:
            raise ValueError("❌ 감정사전이 비어있습니다!")
        
        print(f"   ✅ {len(self.emotion_dict)}개 단어 로드 완료\n")
        
        self.positive_words = set()
        self.negative_words = set()
        self._categorize_emotion_words()
        
        print("감정 단어 우선 정렬기 초기화 완료!\n")
    
    def load_emotion_dict(self, file_path: str) -> Dict[str, float]:
        """
        감정사전 로드
        지원 형식: JSON, CSV, TXT (탭/공백 구분)
        """
        emotion_dict = {}
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    emotion_dict = {k: float(v) for k, v in data.items()}
            
            elif file_ext == '.csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            word = row[0].strip()
                            try:
                                score = float(row[1].strip())
                                emotion_dict[word] = score
                            except ValueError:
                                continue
            
            elif file_ext in ['.txt', '.tsv']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 2:
                            word = parts[0]
                            try:
                                score = float(parts[1])
                                emotion_dict[word] = score
                            except ValueError:
                                continue
            
            else:
                raise ValueError(f"❌ 지원하지 않는 파일 형식: {file_ext}")
            
            return emotion_dict
        
        except Exception as e:
            print(f"❌ 감정사전 로드 실패: {e}")
            raise
    
    def _categorize_emotion_words(self):
        """감정 단어 분류 (긍정/부정)"""
        for word, score in self.emotion_dict.items():
            if score > 0:
                self.positive_words.add(word)
            elif score < 0:
                self.negative_words.add(word)
    
    def load_json_file(self, file_path: str) -> Dict:
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ❌ 파일 읽기 실패 ({file_path}): {e}")
            return {}
    
    def extract_emotion_words(self, nouns: List[str]) -> Dict:
        """
        명사에서 감정 단어만 추출하고 빈도수로 수치화
        단어 자체의 빈도수만 표시 (예: 싫다 5회, 재밌다 3회, 친구 2회)
        """
        emotion_word_freq = Counter()
        
        # 명사 중 감정사전에 있는 단어 추출
        for noun in nouns:
            if noun in self.emotion_dict:
                emotion_word_freq[noun] += 1
        
        # 빈도수 기준 정렬
        sorted_emotions = emotion_word_freq.most_common()
        
        # 긍정/부정 분류
        positive_words = []
        negative_words = []
        
        for word, freq in sorted_emotions:
            score = self.emotion_dict[word]
            if score > 0:
                positive_words.append((word, freq, score))
            elif score < 0:
                negative_words.append((word, freq, score))
        
        # 빈도수 기준 정렬
        positive_words.sort(key=lambda x: x[1], reverse=True)
        negative_words.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_emotion_words': len(emotion_word_freq),
            'emotion_words': sorted_emotions,
            'positive_words': positive_words,
            'negative_words': negative_words,
            'positive_count': sum(freq for _, freq, _ in positive_words),
            'negative_count': sum(freq for _, freq, _ in negative_words),
            'total_emotion_frequency': sum(emotion_word_freq.values())
        }
    
    def analyze_single_file(self, morpheme_filename: str) -> Dict:
        """단일 파일 분석"""
        
        morpheme_path = os.path.join(self.morpheme_folder, morpheme_filename)
        
        if not os.path.exists(morpheme_path):
            print(f"❌ 형태소 파일 없음: {morpheme_path}")
            return None
        
        print(f"\n{'='*70}")
        print(f" 감정 단어 추출 중: {morpheme_filename}")
        print('='*70)
        
        # 형태소 분석 결과 로드
        morpheme_data = self.load_json_file(morpheme_path)
        if not morpheme_data:
            return None
        
        nouns = morpheme_data.get('all_nouns', [])
        
        if not nouns:
            print(f"   ⚠️  명사가 없습니다.")
            return None
        
        print(f"   총 명사 개수: {len(nouns)}개")
        
        # 감정 단어 추출 및 수치화
        result = self.extract_emotion_words(nouns)
        
        print(f"\n   ✅ 감정 단어 추출 완료")
        print(f"      감정 단어 종류: {result['total_emotion_words']}개")
        print(f"      감정 단어 총 빈도: {result['total_emotion_frequency']}회")
        print(f"      긍정 단어: {result['positive_count']}회")
        print(f"      부정 단어: {result['negative_count']}회")
        
        # 상위 감정 단어 출력
        if result['emotion_words']:
            print(f"\n    추출된 감정 단어 (빈도수 기준):")
            for word, freq in result['emotion_words'][:15]:
                score = self.emotion_dict[word]
                polarity = ":)" if score > 0 else ":("
                print(f"      {polarity} {word}: {freq}회")
        
        if result['positive_words']:
            print(f"\n    긍정 단어:")
            for word, freq, score in result['positive_words'][:8]:
                print(f"      {word}: {freq}회")
        
        if result['negative_words']:
            print(f"\n    부정 단어:")
            for word, freq, score in result['negative_words'][:8]:
                print(f"      {word}: {freq}회")
        
        # 결과 저장
        output_filename = Path(morpheme_filename).stem.replace('_morpheme', '_emotion_ranking.json')
        output_path = os.path.join(self.output_folder, output_filename)
        
        # 저장 형식 (시각화를 위해 최적화)
        save_data = {
            'filename': morpheme_filename,
            'total_nouns': len(nouns),
            'emotion_words_count': result['total_emotion_words'],
            'emotion_words_frequency': result['total_emotion_frequency'],
            'positive_frequency': result['positive_count'],
            'negative_frequency': result['negative_count'],
            'emotion_words': result['emotion_words'],
            'positive_words': [[word, freq] for word, freq, score in result['positive_words']],
            'negative_words': [[word, freq] for word, freq, score in result['negative_words']]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n    결과 저장: {output_path}")
        
        return save_data
    
    def analyze_all_files(self) -> List[Dict]:
        """모든 파일 분석"""
        
        morpheme_files = sorted([
            f for f in os.listdir(self.morpheme_folder) 
            if f.endswith('_morpheme.json')
        ])
        
        if not morpheme_files:
            print(f"❌ 형태소 분석 파일 없음: {self.morpheme_folder}")
            return []
        
        print(f"\n 총 {len(morpheme_files)}개 파일 감정 단어 추출 시작")
        
        results = []
        all_emotion_words = Counter()
        all_positive = Counter()
        all_negative = Counter()
        
        for i, morpheme_filename in enumerate(morpheme_files, 1):
            print(f"\n[{i}/{len(morpheme_files)}]")
            
            result = self.analyze_single_file(morpheme_filename)
            
            if result:
                results.append(result)
                
                # 전체 통계 누적
                for word, freq in result['emotion_words']:
                    all_emotion_words[word] += freq
                
                for word, freq in result['positive_words']:
                    all_positive[word] += freq
                
                for word, freq in result['negative_words']:
                    all_negative[word] += freq
        
        # 전체 통계 출력 및 저장
        if results:
            print(f"\n\n{'='*70}")
            print(f" 전체 감정 단어 통계")
            print('='*70)
            
            # 상위 감정 단어
            print(f"\n    전체 추출 감정 단어 (Top 20):")
            for word, freq in all_emotion_words.most_common(20):
                score = self.emotion_dict[word]
                polarity = ":)" if score > 0 else ":("
                print(f"      {polarity} {word}: {freq}회")
            
            # 상위 긍정 단어
            if all_positive:
                print(f"\n    긍정 단어 통계:")
                for word, freq in all_positive.most_common(10):
                    print(f"      {word}: {freq}회")
            
            # 상위 부정 단어
            if all_negative:
                print(f"\n    부정 단어 통계:")
                for word, freq in all_negative.most_common(10):
                    print(f"      {word}: {freq}회")
            
            # 요약 저장
            summary = {
                'total_files': len(results),
                'total_emotion_words_types': len(all_emotion_words),
                'total_emotion_words_frequency': sum(all_emotion_words.values()),
                'total_positive_frequency': sum(all_positive.values()),
                'total_negative_frequency': sum(all_negative.values()),
                'top_emotion_words': all_emotion_words.most_common(30),
                'top_positive_words': all_positive.most_common(15),
                'top_negative_words': all_negative.most_common(15)
            }
            
            summary_path = os.path.join(self.output_folder, 'emotion_ranking_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n    전체 요약 저장: {summary_path}")
        
        print(f"\n{'='*70}")
        print(f"✅ 감정 단어 추출 완료!")
        print('='*70)
        
        return results


def main():
    print("\n 5단계: 감정 단어 우선 정렬")
    
    try:
        # 감정사전 경로 자동 설정
        default_dict_path = "data/sentiment/SentiWord_Dict.txt"
        
        # 기본 경로가 없으면 사용자 입력
        if not os.path.exists(default_dict_path):
            emotion_dict_path = input("\n감정사전 파일 경로 (JSON/CSV/TXT): ").strip()
        else:
            print(f"\n 기본 감정사전 사용: {default_dict_path}")
            emotion_dict_path = default_dict_path
        
        ranker = EmotionWordRanker(emotion_dict_path=emotion_dict_path)
        
        print("\n분석 모드 선택:")
        print("1. 단일 파일 분석")
        print("2. 전체 파일 분석")
        
        choice = input("\n선택 (1-2): ").strip()
        
        if choice == '1':
            filename = input("파일명 (예: EG_001_morpheme.json): ").strip()
            ranker.analyze_single_file(filename)
        elif choice == '2':
            ranker.analyze_all_files()
        else:
            print("❌ 잘못된 선택")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
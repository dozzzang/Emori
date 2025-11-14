"""
3단계: 감정 분석 (BERT 모델 전용)
- BERT 딥러닝 모델만 사용하여 문서의 감정 분류
- 레이블 변환 로직 보강
"""

import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
except ImportError:
    print("❌ transformers 라이브러리가 설치되지 않았습니다. 설치: pip install transformers torch")
    exit()


class SentimentAnalyzer:
    """감정 분석기 - BERT 전용"""
    
    MODEL_NAME = "matthewburke/korean_sentiment" 
    
    def __init__(self, morpheme_folder="output/morpheme", output_folder="output/sentiment"):
        self.morpheme_folder = morpheme_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        print("감정 분석기 초기화 중...")
        self.bert_analyzer = self._load_bert_model()
        
        if not self.bert_analyzer:
            raise Exception("❌ BERT 모델 로드 실패. 분석을 진행할 수 없습니다.")
        
        print("✅ 초기화 완료! (BERT 모델만 사용)\n")
    
    def _load_bert_model(self):
        """BERT 모델 로드"""
        print(f"\n🤖 BERT 모델 ({self.MODEL_NAME}) 로딩 중...")
        try:
            # 파이프라인으로 로드
            analyzer = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME
            )
            return analyzer
        except Exception as e:
            print(f"⚠️  {self.MODEL_NAME} 로드 실패: {e}")
            return None
    
    def analyze_bert_based(self, text):
        """BERT 기반 감정 분석"""
        
        if not self.bert_analyzer: return None
        
        try:
            if len(text) > 500: text = text[:500]
            
            result = self.bert_analyzer(text)[0]
            
            # 레이블 변환 맵 (LABEL_0, 1, 2 또는 POSITIVE/NEGATIVE 등 모든 경우 처리)
            label_map = {
                'POSITIVE': '긍정', 'NEGATIVE': '부정', 'NEUTRAL': '중립',
                'positive': '긍정', 'negative': '부정', 'neutral': '중립',
                'LABEL_0': '부정', 'LABEL_1': '중립', 'LABEL_2': '긍정' # <-- 원시 레이블 매핑
            }
            
            sentiment = label_map.get(result['label'], result['label'])
            confidence = result['score']
            
            return {'method': 'bert', 'sentiment': sentiment, 'confidence': round(confidence, 3)}
        
        except Exception as e:
            print(f"   ⚠️  BERT 분석 실패: {e}")
            return None
    
    def load_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
        
    def analyze_single_file(self, morpheme_filename):
        """단일 파일 감정 분석"""
        
        morpheme_path = os.path.join(self.morpheme_folder, morpheme_filename)
        morpheme_data = self.load_json_file(morpheme_path)
        if not morpheme_data: return None
        
        print(f"\n{'='*60}")
        print(f"📄 BERT 감정 분석 중: {morpheme_filename}")
        print('='*60)
        
        # 원본 텍스트 로드
        txt_filename = morpheme_data.get('filename', '')
        txt_path = os.path.join('data/txt_files', txt_filename)
        
        original_text = ""
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
        
        if not original_text: return None
        
        bert_result = self.analyze_bert_based(original_text)
        
        if not bert_result:
             print(f"   ❌ BERT 분석 실패. 해당 파일을 건너뜁니다.")
             return None
             
        print(f"      감정: {bert_result['sentiment']}")
        print(f"      신뢰도: {bert_result['confidence']}")
        
        output_data = {
            'filename': txt_filename,
            'bert_based': bert_result,
            'text_length': len(original_text)
        }
        
        output_filename = Path(morpheme_filename).stem.replace('_morpheme', '_sentiment.json')
        output_path = os.path.join(self.output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n    결과 저장: {output_path}")
        return output_data
    
    def analyze_all_files(self):
        """전체 파일 분석"""
        morpheme_files = sorted([f for f in os.listdir(self.morpheme_folder) if f.endswith('_morpheme.json')])
        if not morpheme_files: return []
        
        results = []
        for filename in morpheme_files:
            result = self.analyze_single_file(filename)
            if result: results.append(result)
        
        if results:
            from collections import Counter
            bert_sentiments = [r['bert_based']['sentiment'] for r in results if r['bert_based']]
            bert_counts = Counter(bert_sentiments)
            
            summary = {
                'total_files': len(results),
                'method': 'bert_only',
                'bert_distribution': dict(bert_counts),
            }
            summary_path = os.path.join(self.output_folder, 'sentiment_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return results


def main():
    print("\n 3단계: 감정 분석 (BERT 모델 전용)")
    try:
        analyzer = SentimentAnalyzer()
        
        choice = input("\n실행 모드 선택: 1. 단일 파일 분석 / 2. 전체 파일 분석 (1-2): ").strip()
        
        if choice == '1':
            filename = input("파일명 (예: EG_001_morpheme.json): ").strip()
            analyzer.analyze_single_file(filename)
        elif choice == '2':
            analyzer.analyze_all_files()
        else:
            print("❌ 잘못된 선택")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
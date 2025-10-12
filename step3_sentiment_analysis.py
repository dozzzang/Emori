"""
3단계: 감정 분석 (통합 버전)
- 방법 1: KNU 감정사전 기반 (빠름, 규칙 기반)
- 방법 2: BERT 딥러닝 모델 (느림, 정확)
"""

import os
import json
from pathlib import Path
from collections import Counter
import urllib.request
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """감정 분석기 - 사전 기반 + BERT"""
    
    def __init__(self, morpheme_folder="output/morpheme", output_folder="output/sentiment", use_bert=False):
        self.morpheme_folder = morpheme_folder
        self.output_folder = output_folder
        self.use_bert = use_bert
        os.makedirs(output_folder, exist_ok=True)
        
        print("감정 분석기 초기화 중...")
        
        # 1. 감정사전 로드
        self.sentiment_dict = self._load_sentiment_lexicon()
        
        # 2. BERT 모델 로드 (옵션)
        self.bert_analyzer = None
        if use_bert:
            self.bert_analyzer = self._load_bert_model()
        
        print("✅ 초기화 완료!\n")
    
    def _download_lexicon(self):
        """KNU 감정사전 다운로드"""
        url = "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/SentiWord_Dict.txt"
        
        lexicon_dir = "data/sentiment"
        os.makedirs(lexicon_dir, exist_ok=True)
        
        lexicon_path = os.path.join(lexicon_dir, "SentiWord_Dict.txt")
        
        if os.path.exists(lexicon_path):
            return lexicon_path
        
        print(" KNU 감정사전 다운로드 중...")
        try:
            urllib.request.urlretrieve(url, lexicon_path)
            print(f"✅ 다운로드 완료")
            return lexicon_path
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            return None
    
    def _load_sentiment_lexicon(self):
        """감정사전 로드"""
        lexicon_path = self._download_lexicon()
        
        if not lexicon_path:
            print("⚠️  기본 감정단어 사용")
            return self._get_basic_sentiment_dict()
        
        sentiment_dict = {}
        
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            word = parts[0]
                            try:
                                polarity = float(parts[1])
                                sentiment_dict[word] = polarity
                            except:
                                continue
            
            print(f"✅ 감정사전 로드: {len(sentiment_dict)}개 단어")
            return sentiment_dict
        
        except Exception as e:
            print(f"⚠️  감정사전 로드 실패: {e}")
            return self._get_basic_sentiment_dict()
    
    def _get_basic_sentiment_dict(self):
        """기본 감정 단어 사전"""
        return {
            # 긍정
            '좋다': 1.0, '행복하다': 1.0, '편안하다': 1.0, '즐겁다': 1.0,
            '기쁘다': 1.0, '만족스럽다': 1.0, '편하다': 1.0, '재미있다': 1.0,
            '훌륭하다': 1.0, '멋지다': 1.0, '감사하다': 1.0,
            
            # 부정
            '나쁘다': -1.0, '불안하다': -1.0, '슬프다': -1.0, '힘들다': -1.0,
            '우울하다': -1.0, '스트레스': -1.0, '불편하다': -1.0, '답답하다': -1.0,
            '무섭다': -1.0, '걱정되다': -1.0, '짜증나다': -1.0
        }
    
    def _load_bert_model(self):
        """BERT 모델 로드"""
        print("\n🤖 BERT 모델 로딩 중... (최초 실행 시 다운로드)")
        
        try:
            from transformers import pipeline
            
            # 한국어 감정 분석 모델들
            models = [
                "matthewburke/korean_sentiment",  # 추천
                "snunlp/KR-ELECTRA-discriminator",
                "beomi/kcbert-base"
            ]
            
            for model_name in models:
                try:
                    analyzer = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        tokenizer=model_name
                    )
                    print(f"✅ BERT 모델 로드 성공: {model_name}")
                    return analyzer
                except Exception as e:
                    print(f"⚠️  {model_name} 실패: {e}")
                    continue
            
            print("❌ 모든 BERT 모델 로드 실패. 사전 기반만 사용합니다.")
            return None
            
        except ImportError:
            print("❌ transformers 라이브러리가 설치되지 않았습니다.")
            print("   설치: pip install transformers torch")
            return None
    
    def analyze_lexicon_based(self, words):
        """사전 기반 감정 분석"""
        
        scores = []
        positive_words = []
        negative_words = []
        
        for word in words:
            if word in self.sentiment_dict:
                score = self.sentiment_dict[word]
                scores.append(score)
                
                if score > 0:
                    positive_words.append((word, score))
                elif score < 0:
                    negative_words.append((word, score))
        
        if scores:
            avg_score = sum(scores) / len(scores)
            total_score = sum(scores)
        else:
            avg_score = 0
            total_score = 0
        
        # 감정 분류
        if avg_score > 0.1:
            sentiment = "긍정"
        elif avg_score < -0.1:
            sentiment = "부정"
        else:
            sentiment = "중립"
        
        return {
            'method': 'lexicon',
            'sentiment': sentiment,
            'avg_score': round(avg_score, 3),
            'total_score': round(total_score, 3),
            'positive_words': sorted(positive_words, key=lambda x: x[1], reverse=True),
            'negative_words': sorted(negative_words, key=lambda x: x[1]),
            'emotion_word_count': len(scores)
        }
    
    def analyze_bert_based(self, text):
        """BERT 기반 감정 분석"""
        
        if not self.bert_analyzer:
            return None
        
        try:
            # 텍스트가 너무 길면 잘라냄 (BERT는 512 토큰 제한)
            if len(text) > 500:
                text = text[:500]
            
            result = self.bert_analyzer(text)[0]
            
            # label을 한글로 변환
            label_map = {
                'POSITIVE': '긍정',
                'NEGATIVE': '부정',
                'NEUTRAL': '중립',
                'positive': '긍정',
                'negative': '부정',
                'neutral': '중립'
            }
            
            sentiment = label_map.get(result['label'], result['label'])
            confidence = result['score']
            
            return {
                'method': 'bert',
                'sentiment': sentiment,
                'confidence': round(confidence, 3)
            }
        
        except Exception as e:
            print(f"   ⚠️  BERT 분석 실패: {e}")
            return None
    
    def analyze_single_file(self, morpheme_filename):
        """단일 파일 감정 분석"""
        
        morpheme_path = os.path.join(self.morpheme_folder, morpheme_filename)
        
        if not os.path.exists(morpheme_path):
            print(f"❌ 파일 없음: {morpheme_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"📄 감정 분석 중: {morpheme_filename}")
        print('='*60)
        
        # 형태소 분석 결과 로드
        try:
            with open(morpheme_path, 'r', encoding='utf-8') as f:
                morpheme_data = json.load(f)
        except Exception as e:
            print(f"❌ 파일 읽기 실패: {e}")
            return None
        
        # 원본 텍스트 경로 추정
        txt_filename = morpheme_data.get('filename', '')
        txt_path = os.path.join('data/txt_files', txt_filename)
        
        original_text = ""
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
        
        # 모든 단어 수집
        all_words = []
        all_words.extend(morpheme_data.get('all_nouns', []))
        all_words.extend(morpheme_data.get('all_verbs', []))
        all_words.extend(morpheme_data.get('all_adjectives', []))
        
        print(f"   분석할 단어 수: {len(all_words)}개")
        
        # 1. 사전 기반 분석
        print(f"\n    사전 기반 분석 중...")
        lexicon_result = self.analyze_lexicon_based(all_words)
        
        print(f"      감정: {lexicon_result['sentiment']}")
        print(f"      평균 점수: {lexicon_result['avg_score']}")
        print(f"      감정 단어: {lexicon_result['emotion_word_count']}개")
        
        if lexicon_result['positive_words']:
            print(f"\n       긍정 단어 (Top 5):")
            for word, score in lexicon_result['positive_words'][:5]:
                print(f"         {word}: +{score}")
        
        if lexicon_result['negative_words']:
            print(f"\n       부정 단어 (Top 5):")
            for word, score in lexicon_result['negative_words'][:5]:
                print(f"         {word}: {score}")
        
        # 2. BERT 기반 분석 (옵션)
        bert_result = None
        if self.use_bert and original_text:
            print(f"\n    BERT 분석 중...")
            bert_result = self.analyze_bert_based(original_text)
            
            if bert_result:
                print(f"      감정: {bert_result['sentiment']}")
                print(f"      신뢰도: {bert_result['confidence']}")
        
        # 결과 저장
        output_data = {
            'filename': txt_filename,
            'lexicon_based': lexicon_result,
            'bert_based': bert_result,
            'text_length': morpheme_data.get('text_length', 0)
        }
        
        output_filename = Path(morpheme_filename).stem.replace('_morpheme', '_sentiment.json')
        output_path = os.path.join(self.output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n    결과 저장: {output_path}")
        
        return output_data
    
    def analyze_all_files(self):
        """모든 파일 감정 분석"""
        
        morpheme_files = sorted([
            f for f in os.listdir(self.morpheme_folder) 
            if f.endswith('_morpheme.json')
        ])
        
        if not morpheme_files:
            print(f"❌ 형태소 분석 파일 없음: {self.morpheme_folder}")
            print("   먼저 Step 2를 실행하세요!")
            return []
        
        print(f"\n📚 총 {len(morpheme_files)}개 파일 감정 분석 시작")
        print(f"   방법: {'사전 + BERT' if self.use_bert else '사전 기반'}")
        
        results = []
        for i, filename in enumerate(morpheme_files, 1):
            print(f"\n[{i}/{len(morpheme_files)}]")
            result = self.analyze_single_file(filename)
            if result:
                results.append(result)
        
        # 전체 통계
        if results:
            print(f"\n\n{'='*60}")
            print(f" 전체 통계")
            print('='*60)
            
            # 사전 기반 통계
            lexicon_sentiments = [r['lexicon_based']['sentiment'] for r in results]
            lexicon_counts = Counter(lexicon_sentiments)
            
            print(f"\n   [사전 기반] 감정 분포:")
            for sentiment, count in lexicon_counts.items():
                percentage = (count / len(results)) * 100
                print(f"      {sentiment}: {count}개 ({percentage:.1f}%)")
            
            avg_scores = [r['lexicon_based']['avg_score'] for r in results]
            overall_avg = sum(avg_scores) / len(avg_scores)
            print(f"\n   전체 평균 감정 점수: {overall_avg:.3f}")
            
            # BERT 통계
            if self.use_bert:
                bert_sentiments = [
                    r['bert_based']['sentiment'] 
                    for r in results 
                    if r['bert_based']
                ]
                if bert_sentiments:
                    bert_counts = Counter(bert_sentiments)
                    print(f"\n   [BERT 기반] 감정 분포:")
                    for sentiment, count in bert_counts.items():
                        percentage = (count / len(bert_sentiments)) * 100
                        print(f"      {sentiment}: {count}개 ({percentage:.1f}%)")
            
            # 요약 저장
            summary = {
                'total_files': len(results),
                'method': 'lexicon + bert' if self.use_bert else 'lexicon',
                'lexicon_distribution': dict(lexicon_counts),
                'overall_avg_score': round(overall_avg, 3),
                'files': [
                    {
                        'filename': r['filename'],
                        'lexicon_sentiment': r['lexicon_based']['sentiment'],
                        'lexicon_score': r['lexicon_based']['avg_score'],
                        'bert_sentiment': r['bert_based']['sentiment'] if r['bert_based'] else None,
                        'bert_confidence': r['bert_based']['confidence'] if r['bert_based'] else None
                    }
                    for r in results
                ]
            }
            
            summary_path = os.path.join(self.output_folder, 'sentiment_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n    전체 요약 저장: {summary_path}")
        
        print(f"\n{'='*60}")
        print(f"✅ 감정 분석 완료!")
        print('='*60)
        
        return results


def main():
    print("\n 3단계: 감정 분석 (통합)")
    
    print("\n분석 방법 선택:")
    print("1. 사전 기반만 (빠름)")
    print("2. 사전 + BERT (느림, 정확)")
    
    method_choice = input("\n선택 (1-2): ").strip()
    use_bert = (method_choice == '2')
    
    try:
        analyzer = SentimentAnalyzer(use_bert=use_bert)
        
        print("\n분석 모드 선택:")
        print("1. 단일 파일 분석")
        print("2. 전체 파일 분석")
        
        choice = input("\n선택 (1-2): ").strip()
        
        if choice == '1':
            filename = input("파일명 (예: EG_001_morpheme.json): ").strip()
            analyzer.analyze_single_file(filename)
        elif choice == '2':
            analyzer.analyze_all_files()
        else:
            print("❌ 잘못된 선택")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
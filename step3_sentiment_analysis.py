"""
3단계: 감정 분석
KNU 감정사전 기반 감정 분석
"""

import os
import json
from pathlib import Path
from collections import Counter
import urllib.request


class SentimentAnalyzer:
    """감정 분석기 - KNU 감정사전 기반"""
    
    def __init__(self, morpheme_folder="output/morpheme", output_folder="output/sentiment"):
        self.morpheme_folder = morpheme_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        print("감정 분석기 초기화 중...")
        
        # 감정사전 로드
        self.sentiment_dict = self._load_sentiment_lexicon()
        
        print("✅ 초기화 완료!\n")
    
    def _download_lexicon(self):
        """KNU 감정사전 다운로드"""
        url = "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/SentiWord_Dict.txt"
        
        lexicon_dir = "data/sentiment"
        os.makedirs(lexicon_dir, exist_ok=True)
        
        lexicon_path = os.path.join(lexicon_dir, "SentiWord_Dict.txt")
        
        if os.path.exists(lexicon_path):
            return lexicon_path
        
        print("📥 감정사전 다운로드 중...")
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
            print("⚠️  감정사전 없이 진행 (기본 감정단어 사용)")
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
            # 긍정 (+1)
            '좋다': 1.0, '행복하다': 1.0, '편안하다': 1.0, '즐겁다': 1.0,
            '기쁘다': 1.0, '만족스럽다': 1.0, '편하다': 1.0, '재미있다': 1.0,
            
            # 부정 (-1)
            '나쁘다': -1.0, '불안하다': -1.0, '슬프다': -1.0, '힘들다': -1.0,
            '우울하다': -1.0, '스트레스': -1.0, '불편하다': -1.0, '답답하다': -1.0
        }
    
    def analyze_text(self, words):
        """단어 리스트에서 감정 분석"""
        
        scores = []
        positive_words = []
        negative_words = []
        neutral_words = []
        
        for word in words:
            if word in self.sentiment_dict:
                score = self.sentiment_dict[word]
                scores.append(score)
                
                if score > 0:
                    positive_words.append((word, score))
                elif score < 0:
                    negative_words.append((word, score))
                else:
                    neutral_words.append(word)
        
        # 전체 감정 점수
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
            'sentiment': sentiment,
            'avg_score': round(avg_score, 3),
            'total_score': round(total_score, 3),
            'positive_words': sorted(positive_words, key=lambda x: x[1], reverse=True),
            'negative_words': sorted(negative_words, key=lambda x: x[1]),
            'neutral_words': neutral_words,
            'emotion_word_count': len(scores)
        }
    
    def analyze_single_file(self, morpheme_filename):
        """단일 형태소 분석 결과 파일에서 감정 분석"""
        
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
        
        # 모든 단어 수집 (명사, 동사, 형용사)
        all_words = []
        all_words.extend(morpheme_data.get('all_nouns', []))
        all_words.extend(morpheme_data.get('all_verbs', []))
        all_words.extend(morpheme_data.get('all_adjectives', []))
        
        print(f"   분석할 단어 수: {len(all_words)}개")
        
        # 감정 분석
        result = self.analyze_text(all_words)
        
        print(f"\n   📊 감정 분석 결과:")
        print(f"      감정: {result['sentiment']}")
        print(f"      평균 점수: {result['avg_score']}")
        print(f"      총점: {result['total_score']}")
        print(f"      감정 단어: {result['emotion_word_count']}개")
        
        if result['positive_words']:
            print(f"\n   😊 긍정 단어 (Top 5):")
            for word, score in result['positive_words'][:5]:
                print(f"      {word}: +{score}")
        
        if result['negative_words']:
            print(f"\n   😢 부정 단어 (Top 5):")
            for word, score in result['negative_words'][:5]:
                print(f"      {word}: {score}")
        
        # 결과 저장
        output_data = {
            'filename': morpheme_data.get('filename', ''),
            'sentiment': result['sentiment'],
            'avg_score': result['avg_score'],
            'total_score': result['total_score'],
            'emotion_word_count': result['emotion_word_count'],
            'positive_words': result['positive_words'],
            'negative_words': result['negative_words'],
            'text_length': morpheme_data.get('text_length', 0)
        }
        
        output_filename = Path(morpheme_filename).stem.replace('_morpheme', '_sentiment.json')
        output_path = os.path.join(self.output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n   💾 결과 저장: {output_path}")
        
        return output_data
    
    def analyze_all_files(self):
        """모든 형태소 분석 파일 처리"""
        
        morpheme_files = sorted([
            f for f in os.listdir(self.morpheme_folder) 
            if f.endswith('_morpheme.json')
        ])
        
        if not morpheme_files:
            print(f"❌ 형태소 분석 파일 없음: {self.morpheme_folder}")
            print("   먼저 Step 2를 실행하세요!")
            return []
        
        print(f"\n📚 총 {len(morpheme_files)}개 파일 감정 분석 시작")
        
        results = []
        for i, filename in enumerate(morpheme_files, 1):
            print(f"\n[{i}/{len(morpheme_files)}]")
            result = self.analyze_single_file(filename)
            if result:
                results.append(result)
        
        # 전체 통계
        if results:
            print(f"\n\n{'='*60}")
            print(f"📊 전체 통계")
            print('='*60)
            
            sentiments = [r['sentiment'] for r in results]
            sentiment_counts = Counter(sentiments)
            
            print(f"\n   감정 분포:")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(results)) * 100
                print(f"      {sentiment}: {count}개 ({percentage:.1f}%)")
            
            avg_scores = [r['avg_score'] for r in results]
            overall_avg = sum(avg_scores) / len(avg_scores)
            
            print(f"\n   전체 평균 감정 점수: {overall_avg:.3f}")
            
            # 전체 요약 저장
            summary = {
                'total_files': len(results),
                'sentiment_distribution': dict(sentiment_counts),
                'overall_avg_score': round(overall_avg, 3),
                'files': [
                    {
                        'filename': r['filename'],
                        'sentiment': r['sentiment'],
                        'score': r['avg_score']
                    }
                    for r in results
                ]
            }
            
            summary_path = os.path.join(self.output_folder, 'sentiment_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n   💾 전체 요약 저장: {summary_path}")
        
        print(f"\n{'='*60}")
        print(f"✅ 감정 분석 완료!")
        print('='*60)
        
        return results


def main():
    print("\n😊 3단계: 감정 분석")
    
    try:
        analyzer = SentimentAnalyzer()
        
        print("\n1. 단일 파일 분석")
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
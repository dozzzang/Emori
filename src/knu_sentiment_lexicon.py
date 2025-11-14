"""
KNU 한국어 감정사전 다운로드 및 로드
data/sentiment/SentiWord_Dict.txt 에서 확인 가능
"""

import urllib.request
import os

def download_knu_lexicon():
    """KNU 감정사전 다운로드"""
    
    url = "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/SentiWord_Dict.txt"
    
    output_dir = "data/sentiment"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "SentiWord_Dict.txt")

    if os.path.exists(output_path):
        print(f"✅ 감정사전이 이미 존재합니다: {output_path}")
        return output_path
    
    print("📥 KNU 감정사전 다운로드 중...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ 다운로드 완료: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return None

def load_knu_lexicon(file_path):
    """감정사전 로드"""
    
    sentiment_dict = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word = parts[0]
                        # 긍정/부정 극성 (-2 ~ +2)
                        polarity = parts[1]
                        sentiment_dict[word] = polarity
        
        print(f"✅ 감정사전 로드 완료: {len(sentiment_dict)}개 단어")
        return sentiment_dict
    
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return {}

if __name__ == "__main__":
    # 다운로드
    path = download_knu_lexicon()
    
    if path:
        # 로드
        lexicon = load_knu_lexicon(path)
        
        # 테스트
        test_words = ['좋다', '나쁘다', '행복하다', '슬프다']
        print("\n테스트:")
        for word in test_words:
            if word in lexicon:
                print(f"  {word}: {lexicon[word]}")
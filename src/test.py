"""
키워드 추출 테스트
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 테스트 문서들
documents = [
    "VR 체험 상담 스트레스 불안 감소",
    "상담사 대화 편안 기분 좋음",
    "VR 기술 체험 인상적 효과"
]

print("="*60)
print("TF-IDF 키워드 추출 테스트")
print("="*60)

# TF-IDF 벡터라이저
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 단어 목록
feature_names = vectorizer.get_feature_names_out()

# 각 문서의 TF-IDF 점수
for i, doc in enumerate(documents):
    print(f"\n문서 {i+1}: {doc}")
    
    # TF-IDF 점수 추출
    scores = tfidf_matrix[i].toarray()[0]
    
    # 점수와 단어 매칭
    word_scores = [(feature_names[j], scores[j]) 
                   for j in range(len(scores)) if scores[j] > 0]
    
    # 점수 순으로 정렬
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("  상위 키워드:")
    for word, score in word_scores[:5]:
        print(f"    {word}: {score:.3f}")

print("\n" + "="*60)
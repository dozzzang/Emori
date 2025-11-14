"""
4단계: 키워드 추출
TF-IDF와 TextRank 알고리즘을 사용한 핵심 키워드 추출
"""

import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx


class KeywordExtractor:
    """키워드 추출기 - TF-IDF & TextRank"""
    
    def __init__(self, morpheme_folder="output/morpheme", output_folder="output/keywords"):
        self.morpheme_folder = morpheme_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        print("키워드 추출기 초기화 완료!\n")
    
    def load_morpheme_file(self, morpheme_path):
        """형태소 분석 결과 로드"""
        try:
            with open(morpheme_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ❌ 파일 읽기 실패: {e}")
            return None
    
    def extract_tfidf_keywords(self, documents, top_n=10):
        """
        TF-IDF 기반 키워드 추출
        
        Args:
            documents: 문서 리스트 (각 문서는 단어들의 문자열)
            top_n: 추출할 키워드 수
        
        Returns:
            각 문서별 키워드와 점수
        """
        
        if len(documents) < 2:
            print("  ⚠️  TF-IDF는 최소 2개 이상의 문서가 필요합니다.")
            return None
        
        # TF-IDF 벡터라이저
        vectorizer = TfidfVectorizer(
            max_features=1000,  # 최대 1000개 단어
            min_df=1,           # 최소 1개 문서에 등장
            max_df=0.8          # 80% 이상 문서에 등장하는 단어 제외
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            results = []
            
            for i in range(len(documents)):
                scores = tfidf_matrix[i].toarray()[0]
                
                # 점수와 단어 매칭
                word_scores = [
                    (feature_names[j], float(scores[j])) 
                    for j in range(len(scores)) if scores[j] > 0
                ]
                
                # 점수 순으로 정렬
                word_scores.sort(key=lambda x: x[1], reverse=True)
                
                results.append(word_scores[:top_n])
            
            return results
        
        except Exception as e:
            print(f"  ❌ TF-IDF 추출 실패: {e}")
            return None
    
    def extract_textrank_keywords(self, words, top_n=10, window=5):
        """
        TextRank 기반 키워드 추출
        
        Args:
            words: 단어 리스트
            top_n: 추출할 키워드 수
            window: 동시 등장 윈도우 크기
        
        Returns:
            키워드와 점수 리스트
        """
        
        if len(words) < 5:
            print("  ⚠️  TextRank는 최소 5개 이상의 단어가 필요합니다.")
            return None
        
        # 그래프 생성
        graph = nx.Graph()
        
        # 노드 추가 (단어)
        unique_words = list(set(words))
        graph.add_nodes_from(unique_words)
        
        # 엣지 추가 (동시 등장)
        for i in range(len(words)):
            for j in range(i + 1, min(i + window, len(words))):
                if words[i] != words[j]:
                    if graph.has_edge(words[i], words[j]):
                        # 기존 엣지 가중치 증가
                        graph[words[i]][words[j]]['weight'] += 1
                    else:
                        # 새 엣지 추가
                        graph.add_edge(words[i], words[j], weight=1)
        
        # PageRank 알고리즘 적용
        try:
            scores = nx.pagerank(graph, weight='weight')
            
            # 점수 순으로 정렬
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_words[:top_n]
        
        except Exception as e:
            print(f"  ❌ TextRank 실패: {e}")
            return None
    
    def extract_frequency_keywords(self, words, top_n=10):
        """
        빈도 기반 키워드 추출 (기본)
        
        Args:
            words: 단어 리스트
            top_n: 추출할 키워드 수
        
        Returns:
            키워드와 빈도 리스트
        """
        counter = Counter(words)
        return counter.most_common(top_n)
    
    def analyze_single_file(self, morpheme_filename):
        """단일 파일 키워드 추출"""
        
        morpheme_path = os.path.join(self.morpheme_folder, morpheme_filename)
        
        if not os.path.exists(morpheme_path):
            print(f"❌ 파일 없음: {morpheme_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f" 키워드 추출 중: {morpheme_filename}")
        print('='*60)
        
        # 형태소 분석 결과 로드
        morpheme_data = self.load_morpheme_file(morpheme_path)
        if not morpheme_data:
            return None
        
        # 명사만 사용 (키워드는 대부분 명사)
        nouns = morpheme_data.get('all_nouns', [])
        
        print(f"   명사 개수: {len(nouns)}개")
        
        if len(nouns) < 5:
            print(f"   ⚠️  명사가 너무 적습니다.")
            return None
        
        # 1. 빈도 기반
        print(f"\n    빈도 기반 키워드:")
        freq_keywords = self.extract_frequency_keywords(nouns, top_n=10)
        for word, count in freq_keywords[:5]:
            print(f"      {word}: {count}회")
        
        # 2. TextRank
        print(f"\n     TextRank 기반 키워드:")
        textrank_keywords = self.extract_textrank_keywords(nouns, top_n=10)
        if textrank_keywords:
            for word, score in textrank_keywords[:5]:
                print(f"      {word}: {score:.4f}")
        
        # 결과 저장
        output_data = {
            'filename': morpheme_data.get('filename', ''),
            'total_nouns': len(nouns),
            'frequency_keywords': freq_keywords,
            'textrank_keywords': textrank_keywords if textrank_keywords else []
        }
        
        output_filename = Path(morpheme_filename).stem.replace('_morpheme', '_keywords.json')
        output_path = os.path.join(self.output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n    결과 저장: {output_path}")
        
        return output_data
    
    def analyze_all_files(self):
        """모든 파일 키워드 추출"""
        
        morpheme_files = sorted([
            f for f in os.listdir(self.morpheme_folder) 
            if f.endswith('_morpheme.json')
        ])
        
        if not morpheme_files:
            print(f"❌ 형태소 분석 파일 없음: {self.morpheme_folder}")
            return []
        
        print(f"\n 총 {len(morpheme_files)}개 파일 키워드 추출 시작")
        
        results = []
        all_documents = []
        
        # 개별 파일 처리
        for i, filename in enumerate(morpheme_files, 1):
            print(f"\n[{i}/{len(morpheme_files)}]")
            result = self.analyze_single_file(filename)
            
            if result:
                results.append(result)
                
                # TF-IDF를 위한 문서 준비
                morpheme_path = os.path.join(self.morpheme_folder, filename)
                morpheme_data = self.load_morpheme_file(morpheme_path)
                if morpheme_data:
                    nouns = morpheme_data.get('all_nouns', [])
                    # 문서를 하나의 문자열로
                    all_documents.append(' '.join(nouns))
        
        # TF-IDF (전체 문서 대상)
        if len(all_documents) >= 2:
            print(f"\n\n{'='*60}")
            print(f" TF-IDF 키워드 추출 (전체 문서)")
            print('='*60)
            
            tfidf_results = self.extract_tfidf_keywords(all_documents, top_n=10)
            
            if tfidf_results:
                for i, keywords in enumerate(tfidf_results):
                    print(f"\n  문서 {i+1} ({morpheme_files[i]}):")
                    for word, score in keywords[:5]:
                        print(f"    {word}: {score:.4f}")
                
                # TF-IDF 결과 저장
                for i, result in enumerate(results):
                    result['tfidf_keywords'] = tfidf_results[i] if i < len(tfidf_results) else []
                    
                    # 업데이트된 결과 저장
                    output_filename = Path(morpheme_files[i]).stem.replace('_morpheme', '_keywords.json')
                    output_path = os.path.join(self.output_folder, output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 전체 통계
        if results:
            print(f"\n\n{'='*60}")
            print(f" 전체 키워드 통계")
            print('='*60)
            
            # 모든 문서의 빈도 키워드 합산
            all_keywords = []
            for result in results:
                for word, count in result.get('frequency_keywords', []):
                    all_keywords.extend([word] * count)
            
            overall_freq = Counter(all_keywords).most_common(20)
            
            print(f"\n   전체 상위 키워드 (Top 20):")
            for word, count in overall_freq:
                print(f"    {word}: {count}회")
            
            # 요약 저장
            summary = {
                'total_files': len(results),
                'overall_top_keywords': overall_freq
            }
            
            summary_path = os.path.join(self.output_folder, 'keywords_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n   전체 요약 저장: {summary_path}")
        
        print(f"\n{'='*60}")
        print(f"✅ 키워드 추출 완료!")
        print('='*60)
        
        return results


def main():
    print("\n 4단계: 키워드 추출")
    
    try:
        extractor = KeywordExtractor()
        
        print("\n추출 모드 선택:")
        print("1. 단일 파일 추출")
        print("2. 전체 파일 추출")
        
        choice = input("\n선택 (1-2): ").strip()
        
        if choice == '1':
            filename = input("파일명 (예: EG_001_morpheme.json): ").strip()
            extractor.analyze_single_file(filename)
        elif choice == '2':
            extractor.analyze_all_files()
        else:
            print("❌ 잘못된 선택")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
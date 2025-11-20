"""
2단계: 형태소 분석 (Q&A 패턴 필터링) - 부사/감탄사 포함
Q) A) 패턴이 있는 문장만 추출해서 분석
(Windows 호환성을 위해 Mecab -> Okt로 변경)
"""

import os
import json
import re
from pathlib import Path
from collections import Counter
from konlpy.tag import Okt  # Mecab 대신 Okt 사용

class QAMorphemeAnalyzer:
    """Q&A 패턴 필터링 형태소 분석기 (Okt 버전)"""
    
    def __init__(self, txt_folder="data/txt_files", output_folder="output/morpheme"):
        self.txt_folder = txt_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        print("☕ Okt(Open Korean Text) 형태소 분석기 초기화 중...")
        try:
            self.okt = Okt()
            # 워밍업 (첫 실행 시 JVM 로딩으로 느릴 수 있음)
            self.okt.pos("테스트", norm=True, stem=True)
            print("✅ Okt 초기화 성공")
        except Exception as e:
            print(f"❌ Okt 초기화 실패: {e}")
            print("Java 환경변수(JAVA_HOME)를 확인해주세요.")
            raise e
        
        self.stopwords = {
            '것', '수', '등', '및', '약', '또', '이', '그', '저', '제',
            '안', '밖', '위', '아래', '좀', '더', '때', '거', '나', '내'
        }
    
    def load_text_file(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"  ❌ 파일 읽기 실패: {e}")
            return None
    
    def extract_qa_sections(self, text, include_qa_label=False):
        """
        Q&A 패턴이 있는 줄만 추출하거나 제거
        """
        lines = text.split('\n')
        qa_lines = []
        # Q/A 패턴 정규식
        qa_pattern = re.compile(r'^[\s]*(Q|Q\)|Q:|질문|Q：|A|A\)|A:|답변|A：)', re.IGNORECASE)
        
        for line in lines:
            if qa_pattern.match(line.strip()):
                qa_lines.append(line)
        
        if include_qa_label:
            return '\n'.join(qa_lines)
        else:
            cleaned_lines = []
            for line in qa_lines:
                # 앞부분 라벨 제거 (Q), A) 등)
                cleaned = re.sub(r'^[\s]*(Q|Q\)|Q:|질문|Q：|A|A\)|A:|답변|A：)\s*', '', line)
                if cleaned.strip():
                    cleaned_lines.append(cleaned)
            return '\n'.join(cleaned_lines)
    
    def get_qa_statistics(self, text):
        q_count = len(re.findall(r'(Q\)|Q:|질문)', text, re.IGNORECASE))
        a_count = len(re.findall(r'(A\)|A:|답변)', text, re.IGNORECASE))
        return {
            'q_count': q_count,
            'a_count': a_count,
            'total_qa_sections': q_count + a_count
        }

    def extract_morphemes(self, text):
        """형태소 추출 (Okt 태그 사용)"""
        # norm=True: 정규화 (되요 -> 돼요), stem=True: 어간 추출 (합니다 -> 하다)
        pos_tags = self.okt.pos(text, norm=True, stem=True)
        
        nouns = []
        verbs = []
        adjectives = []
        adverbs = []
        interjections = []
        
        for word, pos in pos_tags:
            if word in self.stopwords or len(word) <= 1:
                continue
            
            # Okt 태그 매핑
            if pos == 'Noun':       # 명사
                nouns.append(word)
            elif pos == 'Verb':     # 동사
                verbs.append(word)
            elif pos == 'Adjective': # 형용사
                adjectives.append(word)
            elif pos == 'Adverb':    # 부사
                adverbs.append(word)
            elif pos == 'Exclamation': # 감탄사
                interjections.append(word)
        
        return {
            'all_pos': pos_tags,
            'nouns': nouns,
            'verbs': verbs,
            'adjectives': adjectives,
            'adverbs': adverbs,
            'interjections': interjections
        }
    
    def get_frequency(self, words, top_n=20):
        counter = Counter(words)
        return counter.most_common(top_n)
    
    def analyze_single_file(self, txt_filename, mode='qa_only'):
        txt_path = os.path.join(self.txt_folder, txt_filename)
        
        if not os.path.exists(txt_path):
            print(f"❌ 파일 없음: {txt_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f" 분석 중: {txt_filename}")
        print(f"   모드: {'Q&A 패턴만' if mode == 'qa_only' else '전체 텍스트'}")
        print('='*60)
        
        text = self.load_text_file(txt_path)
        if not text: return None
        
        if mode == 'qa_only':
            analyze_text = self.extract_qa_sections(text, include_qa_label=False)
            if len(analyze_text) == 0:
                analyze_text = text
                mode = 'all'
        else:
            analyze_text = text

        result = self.extract_morphemes(analyze_text)
        
        output_data = {
            'filename': txt_filename,
            'analyzer': 'okt',
            'analysis_mode': mode,
            'original_text_length': len(text),
            'analyzed_text_length': len(analyze_text),
            'qa_statistics': self.get_qa_statistics(text),
            'noun_count': len(result['nouns']),
            'verb_count': len(result['verbs']),
            'adjective_count': len(result['adjectives']),
            'adverb_count': len(result['adverbs']),
            'interjection_count': len(result['interjections']),
            'top_nouns': self.get_frequency(result['nouns'], 10),
            'all_nouns': result['nouns'],
            'all_verbs': result['verbs'],
            'all_adjectives': result['adjectives'],
            'all_adverbs': result['adverbs'],
            'all_interjections': result['interjections']
        }
        
        output_filename = Path(txt_filename).stem + '_morpheme.json'
        output_path = os.path.join(self.output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n    결과 저장: {output_path}")
        return output_data

    def analyze_all_files(self, mode='qa_only'):
        """전체 파일 분석 메서드 (누락된 부분 추가)"""
        txt_files = sorted([f for f in os.listdir(self.txt_folder) if f.endswith('.txt')])
        if not txt_files:
            print(f"❌ 처리할 텍스트 파일이 {self.txt_folder}에 없습니다.")
            return []
        
        print(f"📂 총 {len(txt_files)}개의 파일을 분석합니다.")
        
        results = []
        total_nouns = []
        for i, txt_file in enumerate(txt_files, 1):
            result = self.analyze_single_file(txt_file, mode=mode)
            if result:
                results.append(result)
                total_nouns.extend(result['all_nouns'])
        
        if results:
            summary = {
                'total_files': len(results),
                'analyzer': 'okt',
                'analysis_mode': mode,
                'total_noun_count': len(total_nouns),
                'unique_noun_count': len(set(total_nouns)),
                'top_nouns': self.get_frequency(total_nouns, 50)
            }
            summary_path = os.path.join(self.output_folder, 'morpheme_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 전체 요약 저장 완료: {summary_path}")
            
        return results


# 메인 실행부 (여기가 핵심입니다!)
def main():
    print("\n🔍 2단계: 형태소 분석 (Okt 버전)")
    try:
        analyzer = QAMorphemeAnalyzer()
        
        print("\n분석 모드 선택:")
        print("1. Q&A 패턴만 분석 (권장)")
        print("2. 전체 텍스트 분석")
        mode_choice = input("선택 (1/2): ").strip()
        mode = 'qa_only' if mode_choice == '1' else 'all'
        
        print("\n실행 모드 선택:")
        print("1. 단일 파일 분석")
        print("2. 전체 파일 분석")
        choice = input("선택 (1/2): ").strip()
        
        if choice == '1':
            filename = input("파일명 입력 (예: EG_001.txt): ").strip()
            analyzer.analyze_single_file(filename, mode=mode)
        elif choice == '2':
            analyzer.analyze_all_files(mode=mode)
        else:
            print("❌ 잘못된 선택입니다.")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        # import traceback; traceback.print_exc() # 디버깅용

if __name__ == "__main__":
    main()
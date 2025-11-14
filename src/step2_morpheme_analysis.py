"""
2단계: 형태소 분석 (Q&A 패턴 필터링) - 부사/감탄사 포함
Q) A) 패턴이 있는 문장만 추출해서 분석
"""

import os
import json
import re
from pathlib import Path
from collections import Counter


def init_mecab():
    # ... (Mecab 초기화 로직 생략) ...
    from konlpy.tag import Mecab
    
    paths = [
        '/opt/homebrew/lib/mecab/dic/mecab-ko-dic',  # Apple Silicon
        '/usr/local/lib/mecab/dic/mecab-ko-dic',     # Intel Mac
        None
    ]
    
    for path in paths:
        try:
            if path:
                mecab = Mecab(path)
            else:
                mecab = Mecab()
            
            mecab.morphs("테스트")
            print(f"✅ Mecab 초기화 성공 (경로: {path or '기본'})")
            return mecab
        except:
            continue
    
    raise Exception("Mecab 초기화 실패!")


class QAMorphemeAnalyzer:
    """Q&A 패턴 필터링 형태소 분석기"""
    
    def __init__(self, txt_folder="data/txt_files", output_folder="output/morpheme"):
        self.txt_folder = txt_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        print("Mecab 형태소 분석기 초기화 중...")
        self.mecab = init_mecab()
        
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
        # ... (Q&A 섹션 추출 로직 생략) ...
        lines = text.split('\n')
        qa_lines = []
        qa_pattern = re.compile(r'^[\s]*(Q|Q\)|Q:|질문|Q：|A|A\)|A:|답변|A：)', re.IGNORECASE)
        for line in lines:
            if qa_pattern.match(line.strip()):
                qa_lines.append(line)
        
        if include_qa_label:
            return '\n'.join(qa_lines)
        else:
            cleaned_lines = []
            for line in qa_lines:
                cleaned = re.sub(r'^[\s]*(Q|Q\)|Q:|질문|Q：|A|A\)|A:|답변|A：)\s*', '', line)
                if cleaned.strip():
                    cleaned_lines.append(cleaned)
            return '\n'.join(cleaned_lines)
    
    def get_qa_statistics(self, text):
        # ... (Q&A 통계 로직 생략) ...
        q_count = len(re.findall(r'(Q\)|Q:|질문)', text, re.IGNORECASE))
        a_count = len(re.findall(r'(A\)|A:|답변)', text, re.IGNORECASE))
        
        return {
            'q_count': q_count,
            'a_count': a_count,
            'total_qa_sections': q_count + a_count
        }

    
    def extract_morphemes(self, text):
        """형태소 추출 (명사, 동사, 형용사, 부사, 감탄사 포함)"""
        pos_tags = self.mecab.pos(text)
        
        nouns = []
        verbs = []
        adjectives = []
        adverbs = []
        interjections = []
        
        for word, pos in pos_tags:
            if word in self.stopwords or len(word) <= 1:
                continue
            
            if pos.startswith('NN'):  # 명사
                nouns.append(word)
            elif pos.startswith('VV'):  # 동사
                verbs.append(word)
            elif pos.startswith('VA'):  # 형용사
                adjectives.append(word)
            elif pos.startswith('MA'):  # 부사 (추가)
                adverbs.append(word)
            elif pos.startswith('IC'):  # 감탄사 (추가)
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
        """
        단일 파일 분석
        """
        txt_path = os.path.join(self.txt_folder, txt_filename)
        
        if not os.path.exists(txt_path):
            print(f"❌ 파일 없음: {txt_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f" 분석 중: {txt_filename}")
        print(f"   모드: {'Q&A 패턴만' if mode == 'qa_only' else '전체 텍스트'}")
        print('='*60)
        
        text = self.load_text_file(txt_path)
        if not text:
            return None
        
        # ... (분석 텍스트 선택 로직 생략) ...
        if mode == 'qa_only':
            analyze_text = self.extract_qa_sections(text, include_qa_label=False)
            if len(analyze_text) == 0:
                analyze_text = text
                mode = 'all'
        else:
            analyze_text = text

        result = self.extract_morphemes(analyze_text)
        
        # ... (분석 결과 출력 로직 생략) ...
        
        output_data = {
            'filename': txt_filename,
            'analyzer': 'mecab_extended',
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
        # ... (전체 파일 분석 및 요약 로직 생략) ...
        txt_files = sorted([f for f in os.listdir(self.txt_folder) if f.endswith('.txt')])
        if not txt_files: return []
        
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
                'analyzer': 'mecab_extended',
                'analysis_mode': mode,
                'total_noun_count': len(total_nouns),
                'unique_noun_count': len(set(total_nouns)),
                'top_nouns': self.get_frequency(total_nouns, 50)
            }
            summary_path = os.path.join(self.output_folder, 'morpheme_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
        return results


def main():
    print("\n🔍 2단계: 형태소 분석 (Q&A 패턴 필터링)")
    try:
        analyzer = QAMorphemeAnalyzer()
        
        print("\n분석 모드 선택:")
        mode_choice = input("1. Q&A 패턴만 분석 / 2. 전체 텍스트 분석 (1-2): ").strip()
        mode = 'qa_only' if mode_choice == '1' else 'all'
        
        choice = input("\n실행 모드 선택: 1. 단일 파일 분석 / 2. 전체 파일 분석 (1-2): ").strip()
        
        if choice == '1':
            filename = input("파일명 (예: EG_001.txt): ").strip()
            analyzer.analyze_single_file(filename, mode=mode)
        elif choice == '2':
            analyzer.analyze_all_files(mode=mode)
        else:
            print("❌ 잘못된 선택")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
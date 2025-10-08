"""
2단계: 형태소 분석 (Mecab 경로 자동 감지)
"""

import os
import json
from pathlib import Path
from collections import Counter


def init_mecab():
    """Mecab 초기화 - 경로 자동 감지"""
    from konlpy.tag import Mecab
    
    # 시도할 경로 목록
    paths = [
        '/opt/homebrew/lib/mecab/dic/mecab-ko-dic',  # Apple Silicon
        '/usr/local/lib/mecab/dic/mecab-ko-dic',     # Intel Mac
        None  # 기본 경로
    ]
    
    for path in paths:
        try:
            if path:
                mecab = Mecab(path)
            else:
                mecab = Mecab()
            
            # 테스트
            mecab.morphs("테스트")
            
            print(f"✅ Mecab 초기화 성공 (경로: {path or '기본'})")
            return mecab
        except Exception as e:
            if path:
                print(f"⚠️  경로 {path} 실패")
            continue
    
    raise Exception("Mecab 초기화 실패!")


class MorphemeAnalyzer:
    """형태소 분석기"""
    
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
    
    def extract_morphemes(self, text):
        """형태소 추출"""
        pos_tags = self.mecab.pos(text)
        
        nouns = []
        verbs = []
        adjectives = []
        
        for word, pos in pos_tags:
            if word in self.stopwords or len(word) <= 1:
                continue
            
            if pos.startswith('NN'):  # 명사
                nouns.append(word)
            elif pos.startswith('VV'):  # 동사
                verbs.append(word)
            elif pos.startswith('VA'):  # 형용사
                adjectives.append(word)
        
        return {
            'all_pos': pos_tags,
            'nouns': nouns,
            'verbs': verbs,
            'adjectives': adjectives
        }
    
    def get_frequency(self, words, top_n=20):
        counter = Counter(words)
        return counter.most_common(top_n)
    
    def analyze_single_file(self, txt_filename):
        """단일 파일 분석"""
        txt_path = os.path.join(self.txt_folder, txt_filename)
        
        if not os.path.exists(txt_path):
            print(f"❌ 파일 없음: {txt_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"📄 분석 중: {txt_filename}")
        print('='*60)
        
        text = self.load_text_file(txt_path)
        if not text:
            return None
        
        print(f"   텍스트 길이: {len(text)} 문자")
        
        result = self.extract_morphemes(text)
        
        print(f"\n   📊 분석 결과:")
        print(f"      명사: {len(result['nouns'])}개")
        print(f"      동사: {len(result['verbs'])}개")
        print(f"      형용사: {len(result['adjectives'])}개")
        
        noun_freq = self.get_frequency(result['nouns'], 10)
        
        print(f"\n   🔝 상위 명사 (Top 10):")
        for word, count in noun_freq:
            print(f"      {word}: {count}회")
        
        output_data = {
            'filename': txt_filename,
            'analyzer': 'mecab',
            'text_length': len(text),
            'noun_count': len(result['nouns']),
            'verb_count': len(result['verbs']),
            'adjective_count': len(result['adjectives']),
            'top_nouns': noun_freq,
            'all_nouns': result['nouns'],
            'all_verbs': result['verbs'],
            'all_adjectives': result['adjectives']
        }
        
        output_filename = Path(txt_filename).stem + '_morpheme.json'
        output_path = os.path.join(self.output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n   💾 결과 저장: {output_path}")
        return output_data
    
    def analyze_all_files(self):
        """전체 파일 분석"""
        txt_files = sorted([f for f in os.listdir(self.txt_folder) if f.endswith('.txt')])
        
        if not txt_files:
            print(f"❌ TXT 파일 없음: {self.txt_folder}")
            return []
        
        print(f"\n📚 총 {len(txt_files)}개 파일 분석 시작")
        
        results = []
        for i, txt_file in enumerate(txt_files, 1):
            print(f"\n[{i}/{len(txt_files)}]")
            result = self.analyze_single_file(txt_file)
            if result:
                results.append(result)
        
        # 전체 통계
        if results:
            total_nouns = []
            for result in results:
                total_nouns.extend(result['all_nouns'])
            
            print(f"\n\n{'='*60}")
            print(f"📊 전체 통계")
            print('='*60)
            print(f"\n   전체 명사: {len(total_nouns)}개 (고유: {len(set(total_nouns))}개)")
            
            print(f"\n   🏆 전체 상위 명사 (Top 20):")
            for word, count in self.get_frequency(total_nouns, 20):
                print(f"      {word}: {count}회")
            
            summary = {
                'total_files': len(results),
                'analyzer': 'mecab',
                'total_noun_count': len(total_nouns),
                'unique_noun_count': len(set(total_nouns)),
                'top_nouns': self.get_frequency(total_nouns, 50)
            }
            
            summary_path = os.path.join(self.output_folder, 'morpheme_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n   💾 전체 요약 저장: {summary_path}")
        
        print(f"\n{'='*60}")
        print(f"✅ 분석 완료!")
        print('='*60)
        
        return results


def main():
    print("\n🔍 2단계: 형태소 분석 (Mecab)")
    
    try:
        analyzer = MorphemeAnalyzer()
        
        print("\n1. 단일 파일 분석")
        print("2. 전체 파일 분석")
        
        choice = input("\n선택 (1-2): ").strip()
        
        if choice == '1':
            filename = input("파일명 (예: EG_001.txt): ").strip()
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
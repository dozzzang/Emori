   
import os
import json
from konlpy.tag import Mecab
from pathlib import Path


TXT_DIR = 'data/txt_files'
OUTPUT_DIR = 'output/vr_interview/morpheme'
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MorphemeAnalyzer:
    def __init__(self):
        
        try:
            self.tagger = Mecab()
        except Exception:
            print("🚨 Mecab(은전한닢) 설치가 필요합니다. KoNLPy 설치 가이드를 참고하세요.")
            self.tagger = None

    def analyze_file(self, file_path):
                                                
        if not self.tagger: return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"❌ 파일 읽기 실패: {file_path}, {e}")
            return

        
        morphemes = self.tagger.pos(text)
        
        output_filename = Path(file_path).stem + '_morpheme.json'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        output_data = {
            'filename': Path(file_path).name,
            'file_path': str(Path(file_path).resolve()),
            'total_morphemes': len(morphemes),
            'text_preview': text[:100] + '...'
            
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ Step 2 결과 저장: {output_path}")

def main():
    print("🎯 2단계: LLaMA 분석을 위한 파일 관리 구조 생성")
    
    txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith('.txt')]
    if not txt_files:
        print(f"🛑 {TXT_DIR} 폴더에 분석할 .txt 파일이 없습니다.")
        return

    analyzer = MorphemeAnalyzer()
    
    for filename in txt_files:
        print(f"🔄 분석 중: {filename}")
        analyzer.analyze_file(os.path.join(TXT_DIR, filename))
        
if __name__ == "__main__":
    main()
   
import os
import json
from pathlib import Path


TXT_DIR = 'data/txt_files'
OUTPUT_DIR = 'output/vr_interview/morpheme'
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MorphemeAnalyzer:
    def __init__(self):
        self.use_mecab = False
        self.tagger = None
        
        try:
            from konlpy.tag import Mecab
            self.tagger = Mecab()
            self.use_mecab = True
            print("✅ Mecab 태거 사용 가능")
        except Exception:
            print("⚠️ Mecab을 사용할 수 없습니다. 원본 텍스트를 그대로 사용합니다.")
            self.tagger = None
            self.use_mecab = False

    def analyze_file(self, file_path):
                                                
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"❌ 파일 읽기 실패: {file_path}, {e}")
            return False

        total_morphemes = 0
        
        if self.use_mecab and self.tagger:
            try:
                morphemes = self.tagger.pos(text)
                total_morphemes = len(morphemes)
            except Exception as e:
                print(f"⚠️ 형태소 분석 실패, 원본 텍스트 사용: {e}")
                total_morphemes = len(text.split())
        else:
            total_morphemes = len(text.split())
        
        output_filename = Path(file_path).stem + '_morpheme.json'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        output_data = {
            'filename': Path(file_path).name,
            'file_path': str(Path(file_path).resolve()),
            'total_morphemes': total_morphemes,
            'text_preview': text[:100] + '...'
            
        }

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ Step 2 결과 저장: {output_path}")
            return True
        except Exception as e:
            print(f"❌ 파일 저장 실패: {output_path}, {e}")
            return False

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
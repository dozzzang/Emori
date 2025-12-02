import os
import json
from prompts import get_counseling_prompt
from dotenv import load_dotenv
from step2_morpheme_analysis_window import QAMorphemeAnalyzer
from groq import Groq  

INPUT_FOLDER = "data/txt_files"
OUTPUT_FOLDER = "output/llama3"

def clean_text(text):
    if QAMorphemeAnalyzer:
        analyzer = QAMorphemeAnalyzer()
        
        return analyzer.extract_qa_sections(text, include_qa_label=False)
    else:
        return text
    
def analyze_file(client, filename):
    file_path = os.path.join(INPUT_FOLDER, filename)
    
    
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    if not raw_text.strip():
        return None

    
    cleaned_text = clean_text(raw_text)
    
    
    prompt = get_counseling_prompt(cleaned_text)
    
    print(f"🤖 [Llama-3] 분석 중... {filename}")

    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, 
            temperature=0.5 
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        
        if isinstance(result_json, list):
            result_json = {"analysis_result": result_json}
            
        elif isinstance(result_json, dict) and "analysis_result" not in result_json:
            pass 
        
        result_json['filename'] = filename
        result_json['original_length'] = len(raw_text)
        
        return result_json

    except Exception as e:
        print(f"API 호출 중 오류 발생 ({filename}): {e}")
        return None
    
def main():
    
    load_dotenv()
    client = Groq("GROQ_API_KEY")
    
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"📂 결과 저장 폴더 생성 완료: {OUTPUT_FOLDER}")
    
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"입력 폴더가 없습니다: {INPUT_FOLDER}")
        return

    txt_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    print(f"📄 총 {len(txt_files)}개의 파일\n")
    
    
    success_count = 0
    
    for filename in txt_files:
        result = analyze_file(client, filename)
        
        if result:
            
            output_filename = filename.replace('.txt', '_llama_analysis.json')
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"저장 완료: {output_path}\n")
            success_count += 1
            
    print(f"전체 완료(성공: {success_count}/{len(txt_files)})")

if __name__ == "__main__":
    main()
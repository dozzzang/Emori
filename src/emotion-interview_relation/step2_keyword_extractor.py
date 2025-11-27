import os
import json
from pathlib import Path
from groq import Groq 
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# 🚨 사용자 Groq API Key 설정
GROQ_API_KEY = "" 
LLAMA_MODEL_NAME = "llama-3.1-8b-instant" 

# --- 파일 경로 설정 ---
INTERVIEW_INPUT_DIR = 'data/txt_files'
OUTPUT_DIR = 'output/emotionRelation/interviewEmotion'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- 파일 경로 설정 끝 ---

class KeywordExtractor:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY) 
        try:
            self.sbert_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            print("✅ SBERT 모델 로드 완료!")
        except Exception as e:
            raise Exception(f"❌ SBERT 모델 로드 실패: {e}")

    def get_document_text(self, filename: str):
        """원본 인터뷰 TXT 파일을 로드"""
        txt_path = os.path.join(INTERVIEW_INPUT_DIR, filename)
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        print(f"🛑 파일 로드 실패: '{txt_path}' 경로에 파일이 존재하지 않습니다.")
        return None

    def call_llama3_keyword_analysis(self, interview_text: str):
        # 🚨 강화된 LLaMA 프롬프트 (생략) ...
        system_prompt = (
            "당신은 아동 심리 및 행동 전문가입니다. 다음 인터뷰 전문을 분석하여, "
            "**아동의 주된 감정 기조(긍정/부정)를 강화하거나 설명하는 구체적인 경험, 상황, 심리적 상태**에 집중하세요.\n\n"
            "[핵심 상황 키워드 추출 지침 (Blue Dot 정의)]\n"
            "1. **목표:** 아동이 **만족감, 성취감, 즐거움**을 느꼈던 경험과 관련된 핵심 **상황 키워드**를 최우선으로 추출하세요.\n"
            "2. **키워드 유형:** 명사(사건/대상) 또는 형용사(심리적 상태) 형태여야 합니다.\n"
            "3. **키워드 수:** 최소 25개에서 최대 30개의 키워드를 추출하여, 연관성 측정 대상을 최대한 확보하세요.\n"
            "4. **제외 대상 강화:** 단순한 1차원적인 감정 단어(예: 좋다, 싫다, 기쁘다, 슬프다)는 **절대 추출하지 마세요.**\n"
            "   - 대신, 그 감정을 유발한 **구체적인 원인, 행위, 상황, 또는 복합적인 심리적 상태**를 표현하는 명사/형용사만 추출하세요. (예: '100점 달성', '보석 십자수 완성', '친구와 같은 반')\n"
            "5. **가중치 및 감성:** 각 키워드가 메인 감정과 **같은 정서적 기조(예: 'Happy'와 '만족')**에 기여하는 정도를 가중치로 부여하세요.\n\n"
            "결과를 반드시 다음 JSON 형식으로만 반환하세요. 'contribution_weight'는 0.00부터 1.00 사이여야 합니다.\n\n"
            "JSON 형식:\n"
            "{\n"
            "  \"interview_summary\": \"...\",\n"
            "  \"contextual_keywords\": [\n"
            "    {\n"
            "      \"word\": \"핵심 상황/심리 상태 키워드\",\n"
            "      \"contribution_weight\": 0.xx,\n"
            "      \"sentiment_label\": \"긍정/부정/중립/복합\",\n"
            "      \"reason\": \"...\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        
        user_prompt = f"인터뷰 전문:\n\n{interview_text[:6000]}"
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                model=LLAMA_MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=2048, temperature=0.1, response_format={"type": "json_object"}
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            # 🚨 API 호출 실패 시 구체적인 메시지 출력
            print(f"❌ LLaMA API 호출/파싱 실패: {e}")
            return None

    def analyze_sbert_similarity(self, keywords: list[str]) -> list[dict]:
        # ... (SBERT 분석 로직 생략) ...
        if len(keywords) < 2: return []
        embeddings = self.sbert_model.encode(keywords, convert_to_tensor=True)
        similarity_results = []
        cosine_scores = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                sim_score = float(cosine_scores[i][j].item()) 
                similarity_results.append({
                    "word1": keywords[i], "word2": keywords[j], "sbert_score": round(sim_score, 4) 
                })
        return similarity_results

    def analyze_single_file(self, filename: str, student_name: str):
        original_text = self.get_document_text(filename)
        if not original_text: return
        
        base_filename = student_name
        
        print(f"\n{'='*60}\n✨ 2단계 LLaMA/SBERT 분석 시작 (원본: {filename}, 저장명: {base_filename})\n{'='*60}")
        llama_analysis = self.call_llama3_keyword_analysis(original_text)
        
        if not llama_analysis or 'contextual_keywords' not in llama_analysis:
            print(f"🛑 {base_filename}: LLaMA 분석 결과가 유효하지 않아 저장을 건너뜁니다.")
            return 

        keywords_for_sbert = [item['word'] for item in llama_analysis.get('contextual_keywords', [])]
        sbert_results = self.analyze_sbert_similarity(keywords_for_sbert) 
        
        output_data = {
            'analysis_target': base_filename,
            'contextual_keywords': llama_analysis.get('contextual_keywords', []), 
            'sbert_similarity_analysis': sbert_results, 
        }
        
        output_filename = base_filename + '_interviewEmotion.json'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        try:
            # 🚨 파일 저장 시도
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n  ✅ 최종 Blue Dot JSON 저장 성공: {output_path}")
        except Exception as e:
            # 🚨 파일 저장 실패 시 구체적인 메시지 출력
            print(f"\n🛑 파일 저장 실패: {e}")

def process_all_files(extractor: KeywordExtractor):
    """(기능 단순화를 위해) 이 로직은 추후 전체 파일 처리가 필요할 때 구현하세요."""
    print("현재는 단일 파일 처리만 지원합니다.")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        extractor = KeywordExtractor()
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    print("="*40)
    print("🎯 2단계: LLaMA/SBERT Blue Dot 추출 시작")
    print("="*40)
    
    # 🚨 반복 구조 제거, 단일 선택 후 종료
    choice = input("\n분석 방식을 선택하세요 (1: 단일 파일, 2: 전체 파일, 3: 종료): ")
    
    if choice == '1':
        filename = input("1. 분석할 2단계 인풋 파일명 전체를 입력하세요 (예: MB_004.txt): ")
        student_name = input("2. 저장할 파일명에 사용할 학생 이름을 입력하세요 (예: 홍길동): ") 
        
        if not student_name:
             print("🛑 학생 이름은 필수 입력입니다. 프로그램을 종료합니다.")
             return
             
        extractor.analyze_single_file(filename, student_name)
    elif choice == '2':
        process_all_files(extractor)
    elif choice == '3':
        print("프로그램을 종료합니다.")
    else:
        print("잘못된 입력입니다. 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
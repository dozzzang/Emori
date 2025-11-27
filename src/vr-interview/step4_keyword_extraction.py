"""
step4_llama_sbert_analyzer.py: LLaMA 3.1을 사용한 감성 및 기여도 핵심 분석
"""

import os
import json
from pathlib import Path
from groq import Groq 
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# 🚨 사용자의 Groq API Key 적용됨
GROQ_API_KEY = "사용자의 개인 키 입력 필요"
LLAMA_MODEL_NAME = "llama-3.1-8b-instant" 

# 파일 경로 설정
MORPHEME_DIR = 'output/vr_interview/morpheme'
OUTPUT_DIR = 'output/vr_interview/attention'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class LlamaSbertAnalyzer:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY) 
        
        print(f"🤖 모델 로딩 중: SBERT")
        try:
            # SBERT 임베딩 모델 (유사도 측정용)
            self.sbert_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            print("✅ SBERT 모델 로드 완료!")
        except Exception as e:
            raise Exception(f"❌ 모델 로드 실패: {e}")

    def get_document_text(self, filename):
        """원본 TXT 파일을 로드"""
        txt_path = os.path.join('data/txt_files', filename)
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def call_llama3_analysis(self, interview_text):
        """Groq LLaMA 3.1 API를 호출하여 요청하신 JSON 형식의 결과를 반환"""
        
        system_prompt = (
            "당신은 아동 심리 및 행동 전문가입니다. 다음 인터뷰 전문을 분석하여, "
            "**검사대상의 일상생활 속의 행동, 관계, 갈등 상황**에만 집중하세요.\n\n"
            
            "[중요: 요약 작성 방법]\n"
            "interview_summary 필드에는 반드시 **인터뷰 내용 자체를 요약**해야 합니다.\n"
            "- 검사대상이 말한 구체적인 상황과 경험을 문장으로 풀어서 작성하세요.\n"
            "- (중요)줄 마다 개행 넣고 개조식으로 하고 좀 더 구조화시키세요.\n"
            "- (중요)읽기 좋게, 상황별로 구조화"
            # "- 예시: 'ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ"
            # "ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ'\n"
            "- 최소 5문장 이상으로 작성하세요.\n"
            "- 분석적 해석이 아닌, 인터뷰에서 언급된 실제 내용을 자연스러운 문장으로 요약하세요.\n"
            "- '~이는', '~라고 말함', '~느낌' 등의 표현을 사용하여 아동의 경험을 서술하세요.\n\n"
            
            "[분석 기준]\n"
            "1. **키워드 추출**: 문맥에 가장 중요하게 기여하는 **상황 키워드**를 추출하되, **총 20개**의 키워드를 긍정, 부정, 중립, 복합을 최대한 포함하여 반환하세요.\n"
            "   - 순수 감정 동사나 형용사(좋다, 화나다, 기쁘다)는 제외하고, 감정을 유발하는 **행위나 상황**만 키워드로 추출하세요.\n"
            "2. **가중치 부여**: 각 키워드가 전체 상황 이해에 기여하는 정도(0.0~1.0 사이의 가중치)를 추론하세요.\n"
            "3. **감성 분류**: 각 단어가 기여하는 최종 감성(긍정/부정/중립/복합)을 분류하세요.\n"
            "4. **최종 감성 결정**: 최종 감성 기조는 키워드 가중치 총합이 가장 높은 극성으로 결정되어야 합니다. 수치적 우위를 명확히 설명해야 합니다.\n\n"
            
            "결과를 반드시 다음 JSON 형식으로만 반환하세요. 'confidence'와 'contribution_weight'는 0.00부터 1.00 사이여야 합니다.\n\n"
            "JSON 형식:\n"
            "{\n"
            "  \"primary_sentiment\": \"긍정/부정/중립/복합\",\n"
            "  \"confidence\": 0.85,\n"
            "  \"interview_summary\": \"인터뷰 내용을 자연스러운 문장으로 요약 (최소 5문장 이상, 아동의 경험과 상황을 구체적으로 서술)\",\n"
            "  \"contextual_keywords\": [\n"
            "    {\n"
            "      \"word\": \"키워드\",\n"
            "      \"contribution_weight\": 0.15,\n"
            "      \"sentiment_label\": \"긍정/부정/중립/복합\",\n"
            "      \"reason\": \"이 키워드가 감성에 기여한 근거 (인터뷰 내용 기반)\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        
        user_prompt = f"인터뷰 전문:\n\n{interview_text[:6000]}"
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                model=LLAMA_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            llama_json_string = chat_completion.choices[0].message.content
            result = json.loads(llama_json_string)
            
            # 🚨 interview_summary가 없거나 비어있으면 기본값 추가
            if 'interview_summary' not in result or not result['interview_summary']:
                result['interview_summary'] = '인터뷰 요약을 생성하지 못했습니다.'
            
            return result
        
        except Exception as e:
            print(f"❌ LLaMA API 호출/파싱 실패: {e}")
            return None

    def analyze_sbert_similarity(self, keywords):
        """SBERT 임베딩 유사도를 계산 (차트 간 연결성 분석을 위해 유지)"""
        if len(keywords) < 2: return []
        
        embeddings = self.sbert_model.encode(keywords, convert_to_tensor=True)
        similarity_results = []
        cosine_scores = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                word1 = keywords[i]
                word2 = keywords[j]
                sim_score = float(cosine_scores[i][j].item()) 
                
                similarity_results.append({
                    "word_pair": f"{word1}-{word2}",
                    "word1": word1,
                    "word2": word2,
                    "sbert_score": round(sim_score, 4) 
                })
        return similarity_results

    def analyze_single_file(self, morpheme_filename):
        """단일 파일 분석 및 최종 결과 저장"""
        
        morpheme_path = os.path.join(MORPHEME_DIR, morpheme_filename)
        morpheme_data = self.load_json_file(morpheme_path)
        if not morpheme_data: return
        
        original_text = self.get_document_text(morpheme_data['filename'])
        if not original_text: return

        print(f"\n{'='*60}")
        print(f"✨ LLaMA 3 분석 요청: {morpheme_filename}")
        print('='*60)

        # 1. LLaMA 3 API 호출 및 JSON 데이터 획득
        llama_analysis = self.call_llama3_analysis(original_text)
        
        if not llama_analysis or 'contextual_keywords' not in llama_analysis:
            print("🛑 LLaMA 분석 결과가 유효하지 않습니다.")
            return None 

        # 2. SBERT 유사도 분석
        keywords_for_sbert = [item['word'] for item in llama_analysis.get('contextual_keywords', [])]
        sbert_results = self.analyze_sbert_similarity(keywords_for_sbert) 
        print(f"  ✅ SBERT 유사도 분석 완료. {len(sbert_results)}개 쌍 분석.")

        # 3. 최종 결과 저장 (Step 6의 입력 파일)
        output_data = {
            'filename': morpheme_data['filename'],
            'analysis_source': 'LLaMA3 + SBERT',
            'primary_sentiment': llama_analysis.get('primary_sentiment', '중립'), 
            'confidence': llama_analysis.get('confidence', 0.0),
            'interview_summary': llama_analysis.get('interview_summary', '인터뷰 요약을 찾을 수 없습니다.'),  # 🚨 추가
            'contextual_keywords': llama_analysis.get('contextual_keywords', []), 
            'sbert_similarity_analysis': sbert_results, 
        }
        
        output_filename = Path(morpheme_filename).stem.replace('_morpheme', '_llama_analysis.json')
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n  ✅ 최종 분석 JSON 저장: {output_path}")
        except Exception as e:
            print(f"\n🛑 파일 저장 실패: {e}")
            return None

        print(f"\n  ✅ LLaMA 분석 결과 요약:")
        print(f"     최종 감성: {output_data['primary_sentiment']} (신뢰도: {output_data['confidence']:.2f})")
        print(f"     인터뷰 요약 미리보기: {output_data['interview_summary'][:100]}...")  # 🚨 추가
        
        return output_data
    
    def load_json_file(self, file_path: str) -> dict:
        """JSON 파일 로드 유틸리티"""
        try:
            if not os.path.exists(file_path): return {}
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {}

def main():
    print("\n🎯 4단계: Groq LLaMA 3.1 기반 분석 시작")
    analyzer = LlamaSbertAnalyzer()
    
    morpheme_files = [f for f in os.listdir(MORPHEME_DIR) if f.endswith('_morpheme.json')]
    
    if not morpheme_files:
        print(f"🛑 {MORPHEME_DIR} 폴더에 Step 2 결과 파일이 없습니다. Step 2를 먼저 실행하세요.")
        return

    # 첫 번째 파일만 분석하도록 설정
    filename = input("파일명을 입력하세요 ex) EB_001_morpheme.json : ")
    analyzer.analyze_single_file(filename)


if __name__ == "__main__":
    main()
"""
4단계: BERT Attention Score 추출 및 랭킹
- BERT의 Attention Score를 활용하여 각 단어의 감정 분류 기여도를 측정
- 신뢰도 (confidence) 저장 로직 포함
"""

import os
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import numpy as np

# Step 3와 동일한 모델 이름 재사용
MODEL_NAME = "matthewburke/korean_sentiment" 


class BertAttentionRanker:
    """BERT Attention Score 기반 단어 중요도 추출기"""
    
    def __init__(self, morpheme_folder="output/morpheme", 
                 sentiment_folder="output/sentiment",
                 output_folder="output/attention"):
        
        self.morpheme_folder = morpheme_folder
        self.sentiment_folder = sentiment_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"🤖 BERT Attention 모델 ({MODEL_NAME}) 로딩 중...")
        
        # 🚨 디버깅 코드 추가: 현재 실행 경로 확인
        #print(f"DEBUG: 현재 작업 디렉토리 (CWD): {os.getcwd()}")
        #print(f"DEBUG: 찾는 Sentiment 폴더 경로: {os.path.join(os.getcwd(), sentiment_folder)}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                output_attentions=True
            )
            self.model.eval()
            print("✅ BERT Attention 모델 로드 완료!")
        except Exception as e:
            raise Exception(f"❌ BERT 모델 로드 실패: {e}")

    def load_json_file(self, file_path):
        """파일 경로를 받아 JSON 파일을 로드"""
        try:
            if not os.path.exists(file_path):
                # 파일이 존재하지 않으면 None 반환
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            # 파일은 있지만 JSON 형식이 잘못되었거나 권한 문제가 있을 경우
            print(f"❌ JSON 로드 중 심각한 오류 발생 ({file_path}): {e}")
            return None
        
    def get_document_text(self, filename):
        """원본 TXT 파일을 로드"""
        txt_path = os.path.join('data/txt_files', filename)
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def extract_and_rank(self, morpheme_filename):
        """단일 파일 Attention Score 추출 및 랭킹"""
        
        # 1. 필수 데이터 로드
        morpheme_data = self.load_json_file(os.path.join(self.morpheme_folder, morpheme_filename))
        if not morpheme_data: 
            print(f"⚠️  {morpheme_filename} 파일이 없습니다. Step 2를 먼저 실행하세요.")
            return None
        
        # Sentiment 파일 경로 생성
        sentiment_filename_derived = morpheme_filename.replace('_morpheme.json', '_sentiment.json')
        sentiment_path = os.path.join(self.sentiment_folder, sentiment_filename_derived)
        
        # 🚨 디버깅 코드 추가: 찾는 파일 경로 출력
        #print(f"DEBUG: Sentiment 파일을 찾는 경로: {sentiment_path}") 

        sentiment_data = self.load_json_file(sentiment_path)
        
        if not sentiment_data: 
            # 파일이 없거나 로드에 실패하면 여기서 종료
            print(f"⚠️  {Path(sentiment_path).name} 파일을 찾지 못했습니다. Step 3 실행 및 경로 확인이 필요합니다.")
            return None

        original_text = self.get_document_text(morpheme_data['filename'])
        if not original_text: return None

        print(f"\n{'='*60}")
        print(f"✨ Attention Score 추출 중: {morpheme_filename}")
        print('='*60)

        # 2. 토큰화 및 Attention 추출 (모델 실행)
        inputs = self.tokenizer(original_text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 3. Attention Score 계산 
        attentions = outputs.attentions 
        last_layer_att = attentions[-1][0].mean(dim=0)
        cls_attention = last_layer_att[0, :].cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # 4. 단어별 점수 매핑 및 랭킹
        token_importance = {}
        for i, token in enumerate(tokens[1:-1]):
            word = token.replace('##', '')
            score = float(cls_attention[i+1])
            
            if word not in token_importance:
                token_importance[word] = []
            token_importance[word].append(score)

        # 5. 최종 랭킹
        ranked_words = []
        all_morphemes = morpheme_data.get('all_nouns', []) + morpheme_data.get('all_verbs', []) + morpheme_data.get('all_adjectives', []) + morpheme_data.get('all_adverbs', []) + morpheme_data.get('all_interjections', [])
        
        for word, scores in token_importance.items():
            avg_score = np.mean(scores)
            
            if word in all_morphemes:
                ranked_words.append((word, avg_score))

        ranked_words.sort(key=lambda x: x[1], reverse=True)
        
        # 6. 결과 저장
        bert_result = sentiment_data['bert_based']
        
        output_data = {
            'filename': morpheme_data['filename'],
            'bert_sentiment': bert_result['sentiment'],
            'bert_confidence': bert_result['confidence'], # 신뢰도 저장
            'top_attention_words': ranked_words[:30],
            'total_tokens_analyzed': len(tokens)
        }
        
        output_filename = Path(morpheme_filename).stem.replace('_morpheme', '_attention_rank.json')
        output_path = os.path.join(self.output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n    ✅ Top 10 기여 단어:")
        for word, score in ranked_words[:10]:
             print(f"       {word}: {score:.4f}")
        print(f"\n    결과 저장: {output_path}")

        return output_data
    
    def rank_all_files(self):
        """전체 파일 Attention Score 추출"""
        morpheme_files = sorted([f for f in os.listdir(self.morpheme_folder) if f.endswith('_morpheme.json')])
        if not morpheme_files: return []
        
        results = []
        for filename in morpheme_files:
            result = self.extract_and_rank(filename)
            if result: results.append(result)
        
        return results


def main():
    print("\n 4단계: BERT Attention Score 추출 및 랭킹")
    try:
        ranker = BertAttentionRanker()
        
        choice = input("\n실행 모드 선택: 1. 단일 파일 분석 / 2. 전체 파일 분석 (1-2): ").strip()
        
        if choice == '1':
            # 파일명을 입력할 때 반드시 '.json'까지 포함해야 합니다.
            filename = input("파일명 (예: EG_001_morpheme.json): ").strip()
            ranker.extract_and_rank(filename)
        elif choice == '2':
            ranker.rank_all_files()
        else:
            print("❌ 잘못된 선택")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
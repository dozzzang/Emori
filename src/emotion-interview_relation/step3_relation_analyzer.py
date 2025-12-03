import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


MAIN_EMOTION_DIR = 'output/emotionRelation/mainEmotion'
KEYWORD_INPUT_DIR = 'output/emotionRelation/interviewEmotion'
FINAL_OUTPUT_DIR = 'output/emotionRelation/finalRelation'
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)


SIMILARITY_THRESHOLD = 0.25 


EMOTION_TRANSLATION = {
    
    "Happy": "기쁘다",
    
    
    "Sad": "슬프다",
    
    
    "Angry": "화나다",
    
    
    "fear": "두렵다",
    
    
    "Surprise": "놀라다",
    
    
    "Dislike": "싫다",
}


class RelationAnalyzer:
    def __init__(self):
        try:
            self.sbert_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            print("✅ SBERT 모델 로드 완료!")
        except Exception as e:
            raise Exception(f"❌ SBERT 모델 로드 실패: {e}")

    def load_data(self, base_filename: str):
                                                         
        
        
        emotion_path = os.path.join(MAIN_EMOTION_DIR, base_filename + '.txt') 
        try:
            with open(emotion_path, 'r', encoding='utf-8') as f:
                main_emotion = f.read().strip()
        except FileNotFoundError:
            print(f"🛑 1단계 파일 없음: {emotion_path}")
            return None, None, None
        
        
        keyword_path = os.path.join(KEYWORD_INPUT_DIR, base_filename + '_interviewEmotion.json') 
        try:
            with open(keyword_path, 'r', encoding='utf-8') as f:
                keyword_data = json.load(f)
                keywords = keyword_data.get('contextual_keywords', [])
                intra_relations = keyword_data.get('sbert_similarity_analysis', [])
        except FileNotFoundError:
            print(f"🛑 2단계 파일 없음: {keyword_path}")
            keywords = []
            intra_relations = []
        
        return main_emotion, keywords, intra_relations

    def analyze_inter_node_relation(self, main_emotion: str, keywords: list[dict]) -> list[dict]:
                                                            
        
        if not keywords: 
            return []
        
        
        main_emotion_kr = EMOTION_TRANSLATION.get(main_emotion, main_emotion)
        print(f"\n  📌 감정 단어: {main_emotion} → {main_emotion_kr} (한글 번역 사용)")
        
        keyword_words = [item['word'] for item in keywords]
        all_words = [main_emotion_kr] + keyword_words  
        
        embeddings = self.sbert_model.encode(all_words, convert_to_tensor=True)
        main_embedding = embeddings[0].unsqueeze(0)
        keyword_embeddings = embeddings[1:]
        
        inter_relations = []
        cosine_scores = F.cosine_similarity(main_embedding, keyword_embeddings)
        
        print(f"\n  {'키워드':<15} {'스코어':<10} {'상태':<15}")
        print(f"  {'-'*45}")
        
        for i, keyword_word in enumerate(keyword_words):
            sim_score = float(cosine_scores[i].item())
            
            
            is_connected = sim_score >= SIMILARITY_THRESHOLD
            status = "✅ 연결" if is_connected else "❌ 단절"
            
            print(f"  {keyword_word:<15} {sim_score:<10.4f} {status:<15}")
            
            inter_relations.append({
                "source": main_emotion,
                "target": keyword_word,
                "sbert_score": round(sim_score, 4),
                "is_connected": is_connected,
                "connection_status": "선명 연결" if is_connected else "단절(엉뚱/모름)"
            })
        
        
        connected_count = sum(1 for r in inter_relations if r['is_connected'])
        print(f"\n  📊 연결: {connected_count}개 / 단절: {len(inter_relations) - connected_count}개")
        
        return inter_relations

    def analyze_single_file(self, base_filename: str):
                                 
        
        main_emotion, keywords, intra_relations = self.load_data(base_filename)
        
        if not main_emotion: 
            return None
        
        print(f"\n{'='*60}")
        print(f"✨ 3단계 연관성 분석: {base_filename}")
        print(f"   Black Dot: {main_emotion}")
        print(f"   임계값: {SIMILARITY_THRESHOLD}")
        print('='*60)
        
        inter_relations = self.analyze_inter_node_relation(main_emotion, keywords)
        
        output_data = {
            'analysis_target': base_filename,
            'main_emotion_node': {"word": main_emotion, "color": "Black"},
            'keyword_nodes': keywords,
            'inter_node_relations': inter_relations,
            'intra_node_relations': intra_relations
        }
        
        output_filename = base_filename + '_finalRelation.json'
        output_path = os.path.join(FINAL_OUTPUT_DIR, output_filename)
        
        try:
            os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n  ✅ 결과 저장: {output_path}")
        except Exception as e:
            print(f"\n🛑 저장 실패: {e}")
            return None

    def batch_analyze(self):
                         
        
        if not os.path.exists(MAIN_EMOTION_DIR):
            print(f"🛑 1단계 폴더 없음: {MAIN_EMOTION_DIR}")
            return
        
        emotion_files = [f for f in os.listdir(MAIN_EMOTION_DIR) if f.endswith('.txt')]
        
        if not emotion_files:
            print(f"🛑 1단계 파일 없음 ({MAIN_EMOTION_DIR})")
            return
        
        print(f"\n📂 발견된 파일: {len(emotion_files)}개")
        
        success_count = 0
        for emotion_file in emotion_files:
            base_filename = Path(emotion_file).stem
            result = self.analyze_single_file(base_filename)
            if result is not None:
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"✅ 전체 분석 완료: {success_count}/{len(emotion_files)}개 성공")
        print('='*60)

def main():
    try:
        analyzer = RelationAnalyzer()
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    print('\n' + "="*60)
    print("🎯 3단계: Black-Blue 연관성 분석")
    print("="*60)
    
    while True:
        choice = input("\n선택 (1: 단일 파일, 2: 전체 파일, Q: 종료): ")
        
        if choice == '1':
            filename = input("학생 이름 입력 (예: 김시원): ")
            if filename:
                analyzer.analyze_single_file(filename)
        
        elif choice == '2':
            confirm = input("전체 파일을 분석하시겠습니까? (y/n): ")
            if confirm.lower() == 'y':
                analyzer.batch_analyze()
        
        elif choice.upper() == 'Q':
            print("프로그램을 종료합니다.")
            break
        
        else:
            print("잘못된 입력입니다. 1, 2 또는 Q를 입력하세요.")

if __name__ == "__main__":
    main()
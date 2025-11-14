"""
6단계: 감정 단어 시각화 (BERT Attention 기반)
- BERT의 Attention Score를 단어 중요도로 활용하여 시각화
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import numpy as np


class EmotionVisualizer:
    """감정 단어 시각화기"""
    
    def __init__(self, 
                 attention_folder="output/attention", # Step 4 결과 (Attention)
                 output_folder="output/visualization"):
        self.attention_folder = attention_folder
        self.output_folder = output_folder
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 한글 폰트 설정
        self.font_path = self._setup_korean_font()
        
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print("감정 단어 시각화기 초기화 완료!\n")
    
    def _setup_korean_font(self):
        # ... (한글 폰트 설정 로직 생략) ...
        font_paths_mac = [
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Library/Fonts/NanumGothic.ttf',
        ]
        font_paths_linux = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf',
        ]
        font_paths_windows = [
            'C:\\Windows\\Fonts\\malgun.ttf',
            'C:\\Windows\\Fonts\\arial.ttf',
        ]
        
        all_paths = font_paths_mac + font_paths_linux + font_paths_windows
        
        for font_path in all_paths:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
                return font_path
        
        return None
    
    def load_json_file(self, file_path: str) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {}

    def create_wordcloud_attention(self, attention_rankings: list, bert_sentiment: str, filename: str):
        """BERT Attention Score를 크기로 반영한 워드클라우드"""
        if not attention_rankings: return
        
        word_scores_dict = {word: score for word, score in attention_rankings}
        
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if bert_sentiment == '긍정':
                return "hsl(120, 70%%, %d%%)" % random_state.randint(20, 50)
            elif bert_sentiment == '부정':
                return "hsl(0, 70%%, %d%%)" % random_state.randint(20, 50)
            else:
                return "hsl(0, 0%%, %d%%)" % random_state.randint(30, 60)
        
        wordcloud = WordCloud(
            width=1200, height=800, background_color='white', font_path=self.font_path,
            colormap=None, relative_scaling=0.5, min_font_size=10
        ).generate_from_frequencies(word_scores_dict)
        
        plt.figure(figsize=(14, 10))
        plt.imshow(wordcloud.recolor(color_func=color_func, random_state=3), interpolation='bilinear')
        plt.axis('off')
        plt.title(f'BERT Attention 워드클라우드 (문서 극성: {bert_sentiment})', fontsize=16, fontweight='bold', pad=20)
        
        output_path = os.path.join(self.output_folder, f"{filename}_bert_att_wordcloud.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 워드클라우드 저장: {output_path}")

    def create_bar_chart_attention(self, attention_rankings: list, bert_sentiment: str, filename: str, top_n: int = 15):
        """BERT Attention Score를 막대 길이로 반영한 막대 그래프"""
        if not attention_rankings: return
        
        top_words = attention_rankings[:top_n]
        words = [word for word, score in top_words]
        scores = [score for word, score in top_words]
        
        if bert_sentiment == '긍정':
            color = '#90EE90'
            title_ext = " (긍정 문서)"
        elif bert_sentiment == '부정':
            color = '#FFB6C6'
            title_ext = " (부정 문서)"
        else:
            color = '#B0B0B0'
            title_ext = " (중립 문서)"
            
        plt.figure(figsize=(14, 8))
        bars = plt.barh(range(len(words)), scores, color=color)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Attention Score (기여도)', fontsize=12, fontweight='bold')
        plt.title(f'BERT 기반 단어 기여도 랭킹{title_ext}', fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        
        for i, score in enumerate(scores):
            plt.text(score + 0.001, i, f'{score:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_folder, f"{filename}_bert_att_barchart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 막대 그래프 저장: {output_path}")
        
    def create_pie_chart_for_single_file_bert(self, bert_sentiment: str, confidence: float, filename: str):
        """단일 파일에 대한 BERT 기반 감정 비율 파이 차트 (신뢰도 반영)"""
        if bert_sentiment == '긍정':
            labels, sizes, colors = ['긍정'], [1], ['#90EE90']
        elif bert_sentiment == '부정':
            labels, sizes, colors = ['부정'], [1], ['#FFB6C6']
        else:
            labels, sizes, colors = ['중립'], [1], ['#B0C4DE']
            
        plt.figure(figsize=(10, 8))
        plt.pie(
            sizes, labels=labels, colors=colors, startangle=90,
            # 신뢰도 0.0% 문제를 해결하기 위해 실제 신뢰도 값을 사용
            autopct=lambda p: f'{p:.1f}%\n(신뢰도: {confidence*100:.1f}%)' if p > 0 else '',
            textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}
        )
        plt.title(f'BERT 기반 감정 비율 (문서 극성: {bert_sentiment})', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        output_path = os.path.join(self.output_folder, f"{filename}_bert_pie_single.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 단일 파일 파이 차트 저장: {output_path}")

    def create_comparison_chart_attention(self, attention_rankings: list, bert_sentiment: str, filename: str, top_n: int = 7):
        """BERT 문서 감성 기반 긍정/부정 단어 비교 차트 (Attention Score 반영)"""
        if not attention_rankings: return

        top_words = attention_rankings[:top_n]
        words = [word for word, score in top_words]
        scores = [score for word, score in top_words]

        # BERT 감성 결과에 따라 차트 구성
        pos_words, neg_words, pos_scores, neg_scores = [], [], [], []

        if bert_sentiment == '긍정':
            pos_words, pos_scores = words, scores
            pos_color = '#90EE90'; neg_color = '#FFB6C6'
            pos_label = "긍정 기여도 Top 7"; neg_label = "부정 기여도 Top 7 (없음)"
        elif bert_sentiment == '부정':
            neg_words, neg_scores = words, scores
            pos_color = '#90EE90'; neg_color = '#FFB6C6'
            pos_label = "긍정 기여도 Top 7 (없음)"; neg_label = "부정 기여도 Top 7"
        else:
            pos_words, pos_scores = words, scores
            pos_color = '#B0C4DE'; neg_color = '#B0B0B0'
            pos_label = "중립 문서 기여도 Top 7"; neg_label = "중립 문서 기여도 (없음)"

        fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharex=True)
        fig.suptitle(f'BERT 기반 단어 기여도 비교 (문서 극성: {bert_sentiment})', fontsize=16, fontweight='bold', y=1.02)

        # 긍정 단어 차트
        axes[0].barh(range(len(pos_words)), pos_scores, color=pos_color)
        axes[0].set_yticks(range(len(pos_words)))
        axes[0].set_yticklabels(pos_words, fontsize=11)
        axes[0].set_xlabel('Attention Score', fontsize=12, fontweight='bold')
        axes[0].set_title(pos_label, fontsize=13, fontweight='bold')
        axes[0].invert_yaxis()
        for i, score in enumerate(pos_scores):
            axes[0].text(score + 0.001, i, f'{score:.4f}', va='center', fontsize=10, fontweight='bold')

        # 부정 단어 차트
        axes[1].barh(range(len(neg_words)), neg_scores, color=neg_color)
        axes[1].set_yticks(range(len(neg_words)))
        axes[1].set_yticklabels(neg_words, fontsize=11)
        axes[1].set_xlabel('Attention Score', fontsize=12, fontweight='bold')
        axes[1].set_title(neg_label, fontsize=13, fontweight='bold')
        axes[1].invert_yaxis()
        for i, score in enumerate(neg_scores):
            axes[1].text(score + 0.001, i, f'{score:.4f}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        output_path = os.path.join(self.output_folder, f"{filename}_bert_att_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 긍정/부정 비교 차트 저장: {output_path}")

    
    def visualize_single_file(self, filename_prefix: str):
        """단일 파일 시각화"""
        
        attention_path = os.path.join(self.attention_folder, f'{filename_prefix}_attention_rank.json')
        attention_data = self.load_json_file(attention_path)
        
        if not attention_data:
            print(f"❌ Attention 랭킹 파일 없음: {filename_prefix}")
            return
        
        print(f"\n{'='*70}")
        print(f" 시각화 중: {filename_prefix}")
        print('='*70)
        
        bert_sentiment = attention_data.get('bert_sentiment', '중립')
        bert_confidence = attention_data.get('bert_confidence', 0.0)
        
        # 1. BERT Attention 워드클라우드
        self.create_wordcloud_attention(attention_data['top_attention_words'], bert_sentiment, filename_prefix)
        
        # 2. BERT Attention 막대 그래프
        self.create_bar_chart_attention(attention_data['top_attention_words'], bert_sentiment, filename_prefix, top_n=15)
        
        # 3. 긍정/부정 단어 비교 차트
        self.create_comparison_chart_attention(attention_data['top_attention_words'], bert_sentiment, filename_prefix, top_n=7)
        
        # 4. 단일 파일 파이 차트 (신뢰도 사용)
        self.create_pie_chart_for_single_file_bert(bert_sentiment, bert_confidence, filename=filename_prefix)
        
        print(f"\n   ✅ 시각화 완료!")
    
    def visualize_all_files(self):
        """전체 파일 시각화"""
        attention_files = sorted([f for f in os.listdir(self.attention_folder) if f.endswith('_attention_rank.json')])
        if not attention_files: return
        
        for filename in attention_files:
            file_prefix = Path(filename).stem.replace('_attention_rank', '')
            self.visualize_single_file(file_prefix)

def main():
    print("\n 6단계: 감정 단어 시각화 (BERT Attention 전용)")
    try:
        visualizer = EmotionVisualizer()
        
        choice = input("\n실행 모드 선택: 1. 단일 파일 시각화 / 2. 전체 파일 분석 (1-2): ").strip()
        
        if choice == '1':
            filename_prefix = input("파일 접두사 입력 (예: EB_001): ").strip()
            visualizer.visualize_single_file(filename_prefix)
        elif choice == '2':
            visualizer.visualize_all_files()
        else:
            print("❌ 잘못된 선택")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
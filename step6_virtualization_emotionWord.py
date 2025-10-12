"""
6단계: 감정 단어 시각화
워드클라우드, 막대 그래프, 파이 차트 등으로 감정 단어 시각화
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
                 emotion_ranking_folder="output/emotion_ranking",
                 output_folder="output/visualization"):
        self.emotion_ranking_folder = emotion_ranking_folder
        self.output_folder = output_folder
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 한글 폰트 설정
        self.font_path = self._setup_korean_font()
        
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print("감정 단어 시각화기 초기화 완료!\n")
    
    def _setup_korean_font(self):
        """한글 폰트 설정"""
        # Mac OS
        font_paths_mac = [
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Library/Fonts/NanumGothic.ttf',
        ]
        
        # Linux
        font_paths_linux = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf',
        ]
        
        # Windows
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
        
        print("⚠️  한글 폰트를 찾을 수 없습니다. 기본 폰트로 진행합니다.")
        return None
    
    def load_json_file(self, file_path: str) -> dict:
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 파일 읽기 실패: {e}")
            return {}
    
    def create_wordcloud(self, emotion_words: list, filename: str):
        """
        워드클라우드 생성
        
        Args:
            emotion_words: [["단어", 빈도], ...] 형태
            filename: 저장 파일명
        """
        if not emotion_words:
            print("  ⚠️  감정 단어가 없습니다.")
            return
        
        # 단어와 빈도를 딕셔너리로 변환
        word_freq = {word: freq for word, freq in emotion_words}
        
        # 워드클라우드 생성
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color='white',
            font_path=self.font_path,
            colormap='RdYlGn',  # 빨강(부정) ~ 노랑 ~ 초록(긍정)
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq)
        
        # 저장
        plt.figure(figsize=(14, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('감정 단어 워드클라우드', fontsize=16, fontweight='bold', pad=20)
        
        output_path = os.path.join(self.output_folder, f"{filename}_wordcloud.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 워드클라우드 저장: {output_path}")
    
    def create_bar_chart(self, emotion_words: list, title: str, filename: str, top_n: int = 15):
        """
        막대 그래프 생성
        
        Args:
            emotion_words: [["단어", 빈도], ...] 형태
            title: 그래프 제목
            filename: 저장 파일명
            top_n: 상위 N개만 표시
        """
        if not emotion_words:
            print("  ⚠️  감정 단어가 없습니다.")
            return
        
        # 상위 N개만 추출
        top_words = emotion_words[:top_n]
        words = [word for word, freq in top_words]
        freqs = [freq for word, freq in top_words]
        
        # 색상 설정
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(words)))
        
        # 그래프 생성
        plt.figure(figsize=(14, 8))
        bars = plt.barh(range(len(words)), freqs, color=colors)
        plt.yticks(range(len(words)), words)
        plt.xlabel('빈도수', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()  # 위에서 아래로 정렬
        
        # 막대 위에 숫자 표시
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            plt.text(freq + 0.1, i, str(freq), va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f"{filename}_barchart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 막대 그래프 저장: {output_path}")
    
    def create_pie_chart(self, positive_freq: int, negative_freq: int, filename: str):
        """
        파이 차트 생성 (긍정/부정 비율)
        
        Args:
            positive_freq: 긍정 빈도
            negative_freq: 부정 빈도
            filename: 저장 파일명
        """
        if positive_freq == 0 and negative_freq == 0:
            print("  ⚠️  감정 데이터가 없습니다.")
            return
        
        labels = ['긍정', '부정']
        sizes = [positive_freq, negative_freq]
        colors = ['#90EE90', '#FFB6C6']  # 초록색, 빨강색
        explode = (0.05, 0.05)
        
        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        # 자동 텍스트 스타일 설정
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
        
        plt.title('긍정/부정 감정 비율', fontsize=14, fontweight='bold', pad=20)
        
        # 범례 추가
        plt.legend(
            [f'{label}: {size}회' for label, size in zip(labels, sizes)],
            loc='lower right',
            fontsize=11
        )
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f"{filename}_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 파이 차트 저장: {output_path}")
    
    def create_comparison_chart(self, positive_words: list, negative_words: list, filename: str, top_n: int = 10):
        """
        긍정/부정 단어 비교 차트
        
        Args:
            positive_words: 긍정 단어 리스트
            negative_words: 부정 단어 리스트
            filename: 저장 파일명
            top_n: 상위 N개
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 긍정 단어
        if positive_words:
            top_pos = positive_words[:top_n]
            pos_words = [word for word, freq in top_pos]
            pos_freqs = [freq for word, freq in top_pos]
            
            bars1 = ax1.barh(range(len(pos_words)), pos_freqs, color='#90EE90')
            ax1.set_yticks(range(len(pos_words)))
            ax1.set_yticklabels(pos_words)
            ax1.set_xlabel('빈도수', fontsize=11, fontweight='bold')
            ax1.set_title(' 긍정 단어', fontsize=13, fontweight='bold', pad=15)
            ax1.invert_yaxis()
            
            # 빈도 수치 표시
            for i, (bar, freq) in enumerate(zip(bars1, pos_freqs)):
                ax1.text(freq + 0.1, i, str(freq), va='center', fontsize=10, fontweight='bold')
        
        # 부정 단어
        if negative_words:
            top_neg = negative_words[:top_n]
            neg_words = [word for word, freq in top_neg]
            neg_freqs = [freq for word, freq in top_neg]
            
            bars2 = ax2.barh(range(len(neg_words)), neg_freqs, color='#FFB6C6')
            ax2.set_yticks(range(len(neg_words)))
            ax2.set_yticklabels(neg_words)
            ax2.set_xlabel('빈도수', fontsize=11, fontweight='bold')
            ax2.set_title(' 부정 단어', fontsize=13, fontweight='bold', pad=15)
            ax2.invert_yaxis()
            
            # 빈도 수치 표시
            for i, (bar, freq) in enumerate(zip(bars2, neg_freqs)):
                ax2.text(freq + 0.1, i, str(freq), va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle('긍정/부정 감정 단어 비교', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f"{filename}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 비교 차트 저장: {output_path}")
    
    def visualize_single_file(self, emotion_ranking_filename: str):
        """단일 파일 시각화"""
        
        emotion_path = os.path.join(self.emotion_ranking_folder, emotion_ranking_filename)
        
        if not os.path.exists(emotion_path):
            print(f"❌ 파일 없음: {emotion_path}")
            return
        
        print(f"\n{'='*70}")
        print(f" 시각화 중: {emotion_ranking_filename}")
        print('='*70)
        
        data = self.load_json_file(emotion_path)
        if not data:
            return
        
        file_prefix = Path(emotion_ranking_filename).stem.replace('_emotion_ranking', '')
        
        # 1. 워드클라우드
        emotion_words = data.get('emotion_words', [])
        if emotion_words:
            self.create_wordcloud(emotion_words, file_prefix)
        
        # 2. 전체 감정 단어 막대 그래프
        if emotion_words:
            self.create_bar_chart(
                emotion_words,
                '감정 단어 빈도 분석',
                f'{file_prefix}_all',
                top_n=15
            )
        
        # 3. 긍정/부정 파이 차트
        pos_freq = data.get('positive_frequency', 0)
        neg_freq = data.get('negative_frequency', 0)
        if pos_freq > 0 or neg_freq > 0:
            self.create_pie_chart(pos_freq, neg_freq, file_prefix)
        
        # 4. 긍정/부정 비교 차트
        positive_words = data.get('positive_words', [])
        negative_words = data.get('negative_words', [])
        if positive_words or negative_words:
            self.create_comparison_chart(positive_words, negative_words, file_prefix)
        
        print(f"\n   ✅ 시각화 완료!")
    
    def visualize_all_files(self):
        """모든 파일 시각화"""
        
        emotion_files = sorted([
            f for f in os.listdir(self.emotion_ranking_folder)
            if f.endswith('_emotion_ranking.json') and f != 'emotion_ranking_summary.json'
        ])
        
        if not emotion_files:
            print(f"❌ 감정 단어 파일 없음: {self.emotion_ranking_folder}")
            return
        
        print(f"\n 총 {len(emotion_files)}개 파일 시각화 시작")
        
        for i, filename in enumerate(emotion_files, 1):
            print(f"\n[{i}/{len(emotion_files)}]")
            self.visualize_single_file(filename)
        
        # 전체 요약 시각화
        summary_path = os.path.join(self.emotion_ranking_folder, 'emotion_ranking_summary.json')
        if os.path.exists(summary_path):
            print(f"\n{'='*70}")
            print(f" 전체 요약 시각화")
            print('='*70)
            
            summary = self.load_json_file(summary_path)
            
            # 전체 상위 감정 단어
            top_emotions = summary.get('top_emotion_words', [])
            if top_emotions:
                self.create_bar_chart(
                    top_emotions,
                    '전체 감정 단어 통계 (Top 20)',
                    'summary_all',
                    top_n=20
                )
                self.create_wordcloud(top_emotions, 'summary_all')
            
            # 전체 긍정/부정 비율
            pos_total = summary.get('total_positive_frequency', 0)
            neg_total = summary.get('total_negative_frequency', 0)
            if pos_total > 0 or neg_total > 0:
                self.create_pie_chart(pos_total, neg_total, 'summary')
            
            # 긍정/부정 비교
            pos_top = summary.get('top_positive_words', [])
            neg_top = summary.get('top_negative_words', [])
            if pos_top or neg_top:
                self.create_comparison_chart(pos_top, neg_top, 'summary')
            
            print(f"\n   ✅ 전체 시각화 완료!")
        
        print(f"\n{'='*70}")
        print(f"✅ 모든 시각화 완료!")
        print(f" 저장 위치: {self.output_folder}")
        print('='*70)


def main():
    print("\n 6단계: 감정 단어 시각화")
    
    try:
        visualizer = EmotionVisualizer()
        
        print("\n시각화 모드 선택:")
        print("1. 단일 파일 시각화")
        print("2. 전체 파일 시각화")
        
        choice = input("\n선택 (1-2): ").strip()
        
        if choice == '1':
            filename = input("파일명 (예: EG_001_emotion_ranking.json): ").strip()
            visualizer.visualize_single_file(filename)
        elif choice == '2':
            visualizer.visualize_all_files()
        else:
            print("❌ 잘못된 선택")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
step6_visualizer.py: LLaMA 3 분석 결과를 기반으로 요청된 차트 시각화 및 폴더 정리
- 1. 워드클라우드: 감성별 색상 적용.
- 2. 바 차트: 키워드 15개 이상 수용 가능.
- 3. 파이 차트: 최종 감성 극성 키워드의 가중치 비율로 분할 표시.
- 4. 요약: LLaMA가 생성한 최소 5문장 이상 요약 텍스트 출력.
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud # 🚨 WordCloud 라이브러리 다시 임포트
import numpy as np
from collections import defaultdict
import networkx as nx 

# 파일 경로 설정
ATTENTION_DIR = 'output/attention'
VISUAL_ROOT_DIR = 'output/visualization'
os.makedirs(VISUAL_ROOT_DIR, exist_ok=True)


class EmotionVisualizer:
    
    def __init__(self):
        self.font_path = self._setup_korean_font()
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        print("✅ 감정 단어 시각화기 초기화 완료!\n")
    
    def _setup_korean_font(self):
        """한글 폰트 설정"""
        font_paths_mac = ['/System/Library/Fonts/AppleSDGothicNeo.ttc']
        font_paths_linux = ['/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']
        font_paths_windows = ['C:\\Windows\\Fonts\\malgun.ttf']
        
        all_paths = font_paths_mac + font_paths_linux + font_paths_windows
        
        for font_path in all_paths:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
                return font_path
        
        return None
    
    def load_json_file(self, file_path: str) -> dict:
        """JSON 파일 로드"""
        try:
            if not os.path.exists(file_path): return {}
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            return {}

    def create_summary_txt(self, analysis_data: dict, output_folder: str):
        """인터뷰 요약 내용을 TXT 파일로 생성 (요청 반영: 문장형 요약)"""
        
        primary_sentiment = analysis_data.get('primary_sentiment', '불명')
        confidence = analysis_data.get('confidence', 0.0)
        summary_text = analysis_data.get('interview_summary', 'LLaMA 분석 요약 내용을 찾을 수 없습니다.') 
        
        summary_lines = [
            "==========================================",
            f"🎯 LLaMA 3 아동 심리 분석 보고서: {Path(output_folder).name}",
            "==========================================",
            f"최종 추론 감성 기조: {primary_sentiment} (BERT 기반 신뢰도: {confidence:.3f})",
            "",
            "1. [인터뷰 내용 상세 요약]",
            "------------------------------------------",
            summary_text, 
            "",
            "2. [상황적 키워드 기여 근거]",
            "------------------------------------------"
        ]
        
        keywords = analysis_data.get('contextual_keywords', [])
        for item in keywords:
            word = item.get('word', 'N/A')
            weight = item.get('contribution_weight', 0.0)
            reason = item.get('reason', '근거 없음')
            sentiment_label = item.get('sentiment_label', '중립')
            
            summary_lines.append(f"• 키워드: {word} (분류: {sentiment_label})")
            summary_lines.append(f"  > 기여도: {weight:.4f}")
            summary_lines.append(f"  > 분석 근거: {reason}")
        
        summary_path = os.path.join(output_folder, f"{Path(output_folder).name}_summary.txt")
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(summary_lines))
            print(f"  ✅ [4] 인터뷰 요약 TXT 저장: {Path(summary_path).name}") # 🚨 순서 4로 변경
        except Exception as e:
            print(f"❌ 요약 TXT 파일 저장 실패: {e}")


    def create_wordcloud_chart(self, keywords: list, output_folder: str, filename: str): # 🚨 primary_sentiment 제거
        """
        LLaMA 가중치를 크기로 반영하고, 감성별로 색상을 다르게 표현하는 워드클라우드 (요청 반영)
        """
        if not keywords: return
        
        word_sentiment_map = {item['word']: item['sentiment_label'] for item in keywords}
        word_scores_dict = {item['word']: item['contribution_weight'] for item in keywords}
        
        colors_map = {
            '긍정': '#4CAF50', # 초록색
            '부정': '#F44336', # 빨간색
            '중립': '#9E9E9E', # 회색
            '복합': '#FFC107'  # 주황색 (복합 감성)
        }
        
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            sentiment = word_sentiment_map.get(word, '중립')
            return colors_map.get(sentiment, '#9E9E9E') # 기본값 회색
        
        wordcloud = WordCloud(
            width=1200, height=800, background_color='white', font_path=self.font_path,
            colormap=None, relative_scaling=0.5, min_font_size=10
        ).generate_from_frequencies(word_scores_dict)
        
        plt.figure(figsize=(14, 10))
        # 🚨 color_func 적용
        plt.imshow(wordcloud.recolor(color_func=color_func, random_state=3), interpolation='bilinear') 
        plt.axis('off')
        plt.title(f'LLaMA 기반 키워드 감성별 워드클라우드', fontsize=16, fontweight='bold', pad=20)
        
        output_path = os.path.join(output_folder, f"{filename}_keyword_wordcloud.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ [3] 키워드 감성별 워드클라우드 저장: {Path(output_path).name}") # 🚨 순서 3으로 변경


    def create_contribution_bar_chart(self, keywords: list, primary_sentiment: str, output_folder: str, filename: str):
        """
        감성별 기여도 막대형 차트 (키워드 수량 증가 반영)
        """
        # 1. 감성별로 키워드 분리 및 정렬
        grouped_keywords = defaultdict(list)
        for item in keywords:
            sentiment = item.get('sentiment_label', '중립') 
            item['contribution_weight'] = item.get('contribution_weight', 0.0) 
            grouped_keywords[sentiment].append(item)
        
        for sentiment in grouped_keywords:
            grouped_keywords[sentiment].sort(key=lambda x: x['contribution_weight'], reverse=True)
            
        sentiment_order = ['긍정', '복합', '중립', '부정']
        colors = {'긍정': '#4CAF50', '부정': '#F44336', '중립': '#9E9E9E', '복합': '#FFC107'}
        
        # 2. 서브플롯 생성
        num_charts = len(grouped_keywords)
        if num_charts == 0: return

        fig, axes = plt.subplots(num_charts, 1, figsize=(12, 5.5 * num_charts))
        if num_charts == 1: axes = [axes]
        
        fig.suptitle(f'문서: {filename} | LLaMA 기반 상황적 키워드 기여도 분석', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        max_weight = max(item['contribution_weight'] for item_list in grouped_keywords.values() for item in item_list) if grouped_keywords else 0.1
        
        plot_index = 0
        for sentiment in sentiment_order:
            if sentiment in grouped_keywords:
                data = grouped_keywords[sentiment] 
                words = [item['word'] for item in data]
                weights = [item['contribution_weight'] for item in data]
                reasons = [item.get('reason', '') for item in data]
                
                ax = axes[plot_index]
                bars = ax.barh(words, weights, color=colors[sentiment])
                ax.set_title(f'[{sentiment} 기여도] Top {len(words)} 키워드', fontsize=13)
                ax.set_xlabel('상황적 문맥 기여도 (0.0 ~ 1.0)', fontsize=11)
                ax.invert_yaxis()
                
                ax.set_xlim(right=max_weight * 1.5) 
                
                for bar, weight, reason in zip(bars, weights, reasons):
                    text = f'{weight:.4f}'
                    if reason:
                        reason_wrapped = '\n'.join([reason[i:i+35] for i in range(0, len(reason), 35)])
                        text += f'\n({reason_wrapped})'
                    
                    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                            text, va='center', fontsize=7, ha='left')
                
                plot_index += 1

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        output_path = os.path.join(output_folder, f"{filename}_contribution_barchart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ [1] 감성별 기여도 막대 차트 저장: {Path(output_path).name}")


    def create_sentiment_pie_chart(self, primary_sentiment: str, confidence: float, output_folder: str, filename: str):
        """
        최종 감성과 그 감성에 기여한 키워드별 가중치를 보여주는 원형 차트 (요청 반영)
        """
        keywords = analysis_data.get('contextual_keywords', [])
        
        # 1. 최종 감성(primary_sentiment)과 일치하는 키워드만 필터링
        relevant_keywords = [
            item for item in keywords 
            if item.get('sentiment_label') == primary_sentiment and item.get('contribution_weight', 0) > 0.0
        ]
        
        if not relevant_keywords:
            print(f"🛑 {primary_sentiment}에 기여한 키워드가 없어 파이 차트 생략.")
            return

        # 2. 파이 차트 데이터 준비
        labels = [f"{item['word']} ({item['contribution_weight']:.2f})" for item in relevant_keywords]
        sizes = [item['contribution_weight'] for item in relevant_keywords]
        
        # 3. 색상 설정 (통일된 색상 톤 사용)
        main_color = {'긍정': '#4CAF50', '부정': '#F44336', '중립': '#9E9E9E', '복합': '#FFC107'}.get(primary_sentiment, '#9E9E9E')
        cmap = plt.cm.get_cmap('Greens' if primary_sentiment == '긍정' else 
                                ('Reds' if primary_sentiment == '부정' else 
                                 'coolwarm'))
        
        colors = cmap(np.linspace(0.4, 0.8, len(sizes))) 

        plt.figure(figsize=(10, 8))
        
        wedges, texts, autotexts = plt.pie(
            sizes, labels=labels, colors=colors, startangle=90,
            autopct='%1.1f%%',
            textprops={'fontsize': 10, 'color': 'black'}
        )
        
        plt.text(0, 0, 
                 f'전체 감성: {primary_sentiment}\n(신뢰도: {confidence*100:.1f}%)', 
                 ha='center', va='center', fontsize=12, fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
        
        plt.title(f'최종 감성 "{primary_sentiment}" 기여 키워드 가중치 비율', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        output_path = os.path.join(output_folder, f"{filename}_sentiment_piechart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ [2] 최종 감성 키워드 파이 차트 저장: {Path(output_path).name}")
        

    def visualize_single_file(self, filename_prefix: str):
        """단일 파일 시각화 및 결과 폴더 정리"""
        global analysis_data
        
        analysis_path = os.path.join(ATTENTION_DIR, f'{filename_prefix}_llama_analysis.json')
        analysis_data = self.load_json_file(analysis_path)
        
        if not analysis_data: return
        
        output_folder = os.path.join(VISUAL_ROOT_DIR, filename_prefix)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f" 시각화 중: {filename_prefix}")
        print('='*70)
        
        primary_sentiment = analysis_data.get('primary_sentiment', '중립')
        confidence = analysis_data.get('confidence', 0.0)
        keywords = analysis_data.get('contextual_keywords', [])
        
        # 1. 감성별 기여도 막대 차트
        self.create_contribution_bar_chart(keywords, primary_sentiment, output_folder, filename_prefix)
        
        # 2. 최종 감성 키워드 파이 차트
        self.create_sentiment_pie_chart(primary_sentiment, confidence, output_folder, filename_prefix)
        
        # 3. 키워드 감성별 워드클라우드 (복구 및 색상 적용)
        self.create_wordcloud_chart(keywords, output_folder, filename_prefix) # 🚨 primary_sentiment 제거
        
        # 4. 인터뷰 요약 TXT 생성
        self.create_summary_txt(analysis_data, output_folder)
        
        print(f"\n   ✅ 시각화 완료! 결과 폴더: {output_folder}")
    
    def visualize_all_files(self):
        """전체 파일 시각화"""
        analysis_files = sorted([f for f in os.listdir(ATTENTION_DIR) if f.endswith('_llama_analysis.json')])
        if not analysis_files: 
            print("🛑 분석 파일이 없습니다. Step 4를 먼저 실행하세요.")
            return
        
        print(f"\n🔄 총 {len(analysis_files)}개 파일 시각화 시작...")
        for filename in analysis_files:
            file_prefix = Path(filename).stem.replace('_llama_analysis', '')
            self.visualize_single_file(file_prefix)
        print("\n✅ 전체 시각화 완료.")

def main():
    print("\n 6단계: LLaMA 기반 시각화 시작")
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
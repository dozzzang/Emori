   

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import numpy as np
from collections import defaultdict
import networkx as nx  


ATTENTION_DIR = 'output/vr_interview/attention'
VISUAL_ROOT_DIR = 'output/vr_interview/visualization'
os.makedirs(VISUAL_ROOT_DIR, exist_ok=True)


analysis_data = {}


class EmotionVisualizer:

    def __init__(self):
        self.font_path = self._setup_korean_font()
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        print("✅ 감정 단어 시각화기 초기화 완료!\n")

    def _setup_korean_font(self):
                      
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
                        
        try:
            if not os.path.exists(file_path):
                return {}
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            return {}

    def create_summary_txt(self, analysis_data: dict, output_folder: str):
                                                   

        primary_sentiment = analysis_data.get('primary_sentiment', '불명')
        confidence = analysis_data.get('confidence', 0.0)
        summary_text = analysis_data.get('interview_summary', 'LLaMA 분석 요약 내용을 찾을 수 없습니다.')

        import re
        summary_text = re.sub(r'학생[^은]*은\s*', '', summary_text)
        summary_text = re.sub(r'학생[^가]*가\s*', '', summary_text)
        summary_text = re.sub(r'학생[^를]*를\s*', '', summary_text)
        summary_text = re.sub(r'학생[^의]*의\s*', '', summary_text)
        summary_text = re.sub(r'학생[^에게]*에게\s*', '', summary_text)
        summary_text = re.sub(r'학생\s*', '', summary_text)

        participant_name = Path(output_folder).name
        
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>인터뷰 요약 - {participant_name}</title>
    <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            min-height: 100vh;
            line-height: 1.8;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 16px;
            opacity: 0.9;
            margin-top: 8px;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 22px;
            font-weight: 700;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .sentiment-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            margin-left: 10px;
        }}
        
        .sentiment-positive {{
            background: #4CAF50;
            color: white;
        }}
        
        .sentiment-negative {{
            background: #F44336;
            color: white;
        }}
        
        .sentiment-neutral {{
            background: #9E9E9E;
            color: white;
        }}
        
        .sentiment-complex {{
            background: #FFC107;
            color: #333;
        }}
        
        .summary-text {{
            font-size: 16px;
            line-height: 2;
            color: #444;
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            white-space: pre-wrap;
        }}
        
        .keywords-grid {{
            display: grid;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .keyword-card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .keyword-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .keyword-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .keyword-word {{
            font-size: 18px;
            font-weight: 700;
            color: #333;
        }}
        
        .keyword-weight {{
            font-size: 14px;
            color: #666;
            font-weight: 600;
        }}
        
        .keyword-reason {{
            font-size: 14px;
            color: #555;
            line-height: 1.6;
            margin-top: 8px;
        }}
        
        .confidence-info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            font-size: 14px;
            color: #1976d2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>아동 심리 분석 보고서</h1>
            <div class="subtitle">{participant_name}</div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">
                    최종 감성 기조
                    <span class="sentiment-badge sentiment-{primary_sentiment.lower()}">{primary_sentiment}</span>
                </h2>
                <div class="confidence-info">
                    신뢰도: {confidence:.1%} (BERT 기반 분석)
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">인터뷰 내용 상세 요약</h2>
                <div class="summary-text">{summary_text}</div>
            </div>
            
            <div class="section">
                <h2 class="section-title">상황적 키워드 기여 근거</h2>
                <div class="keywords-grid">
"""

        import re
        keywords = analysis_data.get('contextual_keywords', [])
        for item in keywords:
            word = item.get('word', 'N/A')
            weight = item.get('contribution_weight', 0.0)
            reason = item.get('reason', '근거 없음')
            
            cleaned_reason = re.sub(r'학생[^은]*은\s*', '', reason)
            cleaned_reason = re.sub(r'학생[^가]*가\s*', '', cleaned_reason)
            cleaned_reason = re.sub(r'학생[^를]*를\s*', '', cleaned_reason)
            cleaned_reason = re.sub(r'학생[^의]*의\s*', '', cleaned_reason)
            cleaned_reason = re.sub(r'학생\s*', '', cleaned_reason)
            
            sentiment_label = item.get('sentiment_label', '중립')
            
            sentiment_class = sentiment_label.lower()
            if sentiment_label == '긍정':
                sentiment_class = 'positive'
            elif sentiment_label == '부정':
                sentiment_class = 'negative'
            elif sentiment_label == '중립':
                sentiment_class = 'neutral'
            elif sentiment_label == '복합':
                sentiment_class = 'complex'
            
            html_content += f"""
                    <div class="keyword-card">
                        <div class="keyword-header">
                            <span class="keyword-word">{word}</span>
                            <span class="sentiment-badge sentiment-{sentiment_class}">{sentiment_label}</span>
                        </div>
                        <div class="keyword-weight">기여도: {weight:.3f}</div>
                        <div class="keyword-reason">{cleaned_reason}</div>
                    </div>
"""

        html_content += """
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

        summary_path = os.path.join(output_folder, f"{Path(output_folder).name}_summary.html")

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"  ✅ [4] 인터뷰 요약 HTML 저장: {Path(summary_path).name}")
        except Exception as e:
            print(f"❌ 요약 HTML 파일 저장 실패: {e}")

    
    def create_wordcloud_chart(self, keywords: list, output_folder: str, filename: str):
                   
        if not keywords:
            return

        
        agg = {}
        for item in keywords:
            word = item.get('word')
            if not word:
                continue
            weight = float(item.get('contribution_weight', 0.0))
            label = item.get('sentiment_label', '중립')

            if word in agg:
                agg[word]['weight'] += weight
                if weight > agg[word]['max_weight']:
                    agg[word]['label'] = label
                    agg[word]['max_weight'] = weight
            else:
                agg[word] = {
                    'weight': weight,
                    'label': label,
                    'max_weight': weight
                }

        if not agg:
            return

        
        sorted_words = sorted(
            agg.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        TOP_N = 80
        sorted_words = sorted_words[:TOP_N]

        word_scores_dict = {w: info['weight'] for w, info in sorted_words}
        word_sentiment_map = {w: info['label'] for w, info in sorted_words}

        
        colors_map = {
            '긍정': '#4CAF50',
            '부정': '#F44336',
            '중립': '#9E9E9E',
            '복합': '#FFC107'
        }

        def color_func(word, font_size, position, orientation,
                       random_state=None, **kwargs):
            sentiment = word_sentiment_map.get(word, '중립')
            return colors_map.get(sentiment, '#9E9E9E')

        
        width, height = 800, 800
        mask = np.full((height, width), 255, dtype=np.uint8)  
        center = (width // 2, height // 2)

        
        padding = 180  
        radius = min(width, height) // 2 - padding

        y, x = np.ogrid[:height, :width]
        circle_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
        mask[circle_area] = 0   

        
        wordcloud = WordCloud(
            background_color='white',
            font_path=self.font_path,
            mask=mask,
            width=width,
            height=height,
            
            max_font_size=80,     
            min_font_size=20,     
            max_words=len(word_scores_dict),
            relative_scaling=0.4,  
            prefer_horizontal=1.0,
            random_state=42,
            collocations=False,    
            
        ).generate_from_frequencies(word_scores_dict)

        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            wordcloud.recolor(color_func=color_func, random_state=42),
            interpolation='bilinear'
        )
        ax.axis('off')

        ax.set_title(
            'LLaMA 기반 키워드 감성별 워드클라우드',
            fontsize=16,
            fontweight='bold',
            pad=15
        )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', label='긍정'),
            Patch(facecolor='#FFC107', label='복합'),
            Patch(facecolor='#9E9E9E', label='중립'),
            Patch(facecolor='#F44336', label='부정')
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True
        )

        plt.tight_layout()
        output_path = os.path.join(output_folder, f"{filename}_keyword_wordcloud.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  ✅ [3] 키워드 감성별 워드클라우드 저장: {Path(output_path).name}")


    def create_contribution_bar_chart_html(self, keywords: list, primary_sentiment: str,
                                      output_folder: str, filename: str):
                   
        
        grouped_keywords = defaultdict(list)
        for item in keywords:
            sentiment = item.get('sentiment_label', '중립')
            item['contribution_weight'] = item.get('contribution_weight', 0.0)
            grouped_keywords[sentiment].append(item)

        for sentiment in grouped_keywords:
            grouped_keywords[sentiment].sort(
                key=lambda x: x['contribution_weight'], reverse=True
            )

        sentiment_order = ['긍정', '복합', '중립', '부정']
        colors = {
            '긍정': '#4CAF50',
            '부정': '#F44336',
            '중립': '#9E9E9E',
            '복합': '#FFC107'
        }

        if len(grouped_keywords) == 0:
            return

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("⚠️ Plotly가 설치되지 않았습니다. pip install plotly로 설치해주세요.")
            return

        num_charts = len(grouped_keywords)
        fig = make_subplots(
            rows=num_charts, cols=1,
            subplot_titles=[f'[{sentiment} 기여도] Top {len(grouped_keywords[sentiment])} 키워드' 
                          for sentiment in sentiment_order if sentiment in grouped_keywords],
            vertical_spacing=0.15,
            row_heights=[1.0] * num_charts
        )

        max_weight = max(
            item['contribution_weight']
            for item_list in grouped_keywords.values()
            for item in item_list
        ) if grouped_keywords else 0.1

        plot_index = 1
        for sentiment in sentiment_order:
            if sentiment in grouped_keywords:
                data = grouped_keywords[sentiment]
                words = [item['word'] for item in data]
                weights = [item['contribution_weight'] for item in data]
                reasons = [item.get('reason', '') for item in data]

                import re
                hover_texts = []
                for word, weight, reason in zip(words, weights, reasons):
                    hover_text = f"<b>{word}</b><br>기여도: {weight:.3f}"
                    if reason:
                        cleaned_reason = re.sub(r'학생[^은]*은\s*', '', reason)
                        cleaned_reason = re.sub(r'학생[^가]*가\s*', '', cleaned_reason)
                        cleaned_reason = re.sub(r'학생[^를]*를\s*', '', cleaned_reason)
                        cleaned_reason = re.sub(r'학생[^의]*의\s*', '', cleaned_reason)
                        cleaned_reason = re.sub(r'학생\s*', '', cleaned_reason)
                        hover_text += f"<br>근거: {cleaned_reason[:100]}{'...' if len(cleaned_reason) > 100 else ''}"
                    hover_texts.append(hover_text)

                fig.add_trace(
                    go.Bar(
                        y=words,
                        x=weights,
                        orientation='h',
                        name=sentiment,
                        marker_color=colors[sentiment],
                        text=[f'{w:.3f}' for w in weights],
                        textposition='outside',
                        hovertext=hover_texts,
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=plot_index, col=1
                )

                fig.update_xaxes(
                    title_text="상황적 문맥 기여도 (0.0 ~ 1.0)",
                    range=[0, max_weight * 1.2],
                    row=plot_index, col=1
                )
                fig.update_yaxes(autorange="reversed", row=plot_index, col=1)

                plot_index += 1

        fig.update_layout(
            height=400 * num_charts,
            font={'family': 'Pretendard, sans-serif', 'size': 12},
            template='plotly_white'
        )

        output_path = os.path.join(output_folder, f"{filename}_contribution_barchart.html")
        fig.write_html(output_path, config={'displayModeBar': True})
        print(f"  ✅ [1] 감성별 기여도 막대 차트 HTML 저장: {Path(output_path).name}")

    def create_sentiment_pie_chart(self, primary_sentiment: str, confidence: float,
                                   output_folder: str, filename: str):
                   
        global analysis_data
        keywords = analysis_data.get('contextual_keywords', [])

        
        relevant_keywords = [
            item for item in keywords
            if item.get('sentiment_label') == primary_sentiment
            and item.get('contribution_weight', 0) > 0.0
        ]

        if not relevant_keywords:
            print(f"🛑 {primary_sentiment}에 기여한 키워드가 없어 파이 차트 생략.")
            return

        labels = [
            f"{item['word']} ({item['contribution_weight']:.2f})"
            for item in relevant_keywords
        ]
        sizes = [item['contribution_weight'] for item in relevant_keywords]

        cmap = plt.cm.get_cmap(
            'Greens' if primary_sentiment == '긍정'
            else ('Reds' if primary_sentiment == '부정' else 'coolwarm')
        )
        colors = cmap(np.linspace(0.4, 0.8, len(sizes)))

        plt.figure(figsize=(10, 8))

        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            autopct='%1.1f%%',
            textprops={'fontsize': 10, 'color': 'black'}
        )

        plt.text(
            0, 0,
            f'전체 감성: {primary_sentiment}\n(신뢰도: {confidence * 100:.1f}%)',
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
            color='black',
            bbox=dict(
                facecolor='white',
                alpha=0.7,
                edgecolor='none',
                boxstyle='round,pad=0.5'
            )
        )

        plt.title(
            f'최종 감성 \"{primary_sentiment}\" 기여 키워드 가중치 비율',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        plt.tight_layout()

        output_path = os.path.join(output_folder, f"{filename}_sentiment_piechart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ [2] 최종 감성 키워드 파이 차트 저장: {Path(output_path).name}")

    def visualize_single_file(self, filename_prefix: str):
                                  
        global analysis_data

        analysis_path = os.path.join(
            ATTENTION_DIR, f'{filename_prefix}_llama_analysis.json'
        )
        analysis_data = self.load_json_file(analysis_path)

        if not analysis_data:
            return

        output_folder = os.path.join(VISUAL_ROOT_DIR, filename_prefix)
        os.makedirs(output_folder, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f" 시각화 중: {filename_prefix}")
        print('=' * 70)

        primary_sentiment = analysis_data.get('primary_sentiment', '중립')
        confidence = analysis_data.get('confidence', 0.0)
        keywords = analysis_data.get('contextual_keywords', [])

        
        self.create_contribution_bar_chart_html(
            keywords, primary_sentiment, output_folder, filename_prefix
        )

        
        self.create_sentiment_pie_chart(
            primary_sentiment, confidence, output_folder, filename_prefix
        )

        
        self.create_wordcloud_chart(keywords, output_folder, filename_prefix)

        
        self.create_summary_txt(analysis_data, output_folder)

        print(f"\n   ✅ 시각화 완료! 결과 폴더: {output_folder}")

    def visualize_all_files(self):
                       
        analysis_files = sorted([
            f for f in os.listdir(ATTENTION_DIR)
            if f.endswith('_llama_analysis.json')
        ])
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

        choice = input(
            "\n실행 모드 선택: 1. 단일 파일 시각화 / 2. 전체 파일 분석 (1-2): "
        ).strip()

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

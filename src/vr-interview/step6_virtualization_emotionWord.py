"""
step6_visualizer.py: LLaMA 3 분석 결과를 기반으로 요청된 차트 시각화 및 폴더 정리
- 1. 워드클라우드: 감성별 색상 적용. (원형 마스크 적용, 중앙 밀집/원형 채움 스타일 최종 최적화)
- 2. 바 차트: 키워드 15개 이상 수용 가능.
- 3. 파이 차트: 최종 감성 극성 키워드의 가중치 비율로 분할 표시.
- 4. 요약: LLaMA가 생성한 최소 5문장 이상 요약 텍스트 출력.
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import numpy as np
from collections import defaultdict
import networkx as nx  # (현재는 안 쓰지만 원본 구조 유지)

# 파일 경로 설정
ATTENTION_DIR = 'output/vr_interview/attention'
VISUAL_ROOT_DIR = 'output/vr_interview/visualization'
os.makedirs(VISUAL_ROOT_DIR, exist_ok=True)

# 파이 차트에서 접근하기 위한 전역 분석 데이터
analysis_data = {}


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
            if not os.path.exists(file_path):
                return {}
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
            print(f"  ✅ [4] 인터뷰 요약 TXT 저장: {Path(summary_path).name}")
        except Exception as e:
            print(f"❌ 요약 TXT 파일 저장 실패: {e}")

    # ★ 여기서부터 수정된 워드클라우드 메서드 전체 ★
    def create_wordcloud_chart(self, keywords: list, output_folder: str, filename: str):
        """
        - 같은 단어는 1번만 출력
        - 단어 크기는 contribution_weight 합으로 결정
        - WordCloud mask를 '원 안'이 0, '원 밖'이 255 가 되도록 설정해서
          단어가 원 안에만 배치되도록 함
        - 원 영역을 더 작게 잡고, 글자 크기 범위를 줄여서
          단어들이 더 촘촘하고 원형에 가깝게 모이도록 튜닝
        """
        if not keywords:
            return

        # 1) 단어별로 weight 합산 + 대표 감성(label) 선택 (중복 제거)
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

        # 2) 상위 N개만 사용
        sorted_words = sorted(
            agg.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        TOP_N = 80
        sorted_words = sorted_words[:TOP_N]

        word_scores_dict = {w: info['weight'] for w, info in sorted_words}
        word_sentiment_map = {w: info['label'] for w, info in sorted_words}

        # 감성별 색상
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

        # 3) 원형 마스크 생성 (단어가 들어갈 영역 = 0, 밖 = 255)
        width, height = 800, 800
        mask = np.full((height, width), 255, dtype=np.uint8)  # 기본값 255 (금지 영역)
        center = (width // 2, height // 2)

        # 🟢 원을 더 작게 만들어 단어들이 더 모이도록 (padding ↑)
        padding = 180  # 값이 클수록 원이 작아져서 더 촘촘해짐
        radius = min(width, height) // 2 - padding

        y, x = np.ogrid[:height, :width]
        circle_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
        mask[circle_area] = 0   # 0 = 단어 허용 영역 (원 안)

        # 4) 워드클라우드 생성
        wordcloud = WordCloud(
            background_color='white',
            font_path=self.font_path,
            mask=mask,
            width=width,
            height=height,
            # 글자 크기 범위를 좁혀서 더 빽빽하게
            max_font_size=80,     # 이전보다 살짝 줄임
            min_font_size=20,     # 최소 크기 조금 키워서 전체 볼륨 유지
            max_words=len(word_scores_dict),
            relative_scaling=0.4,  # 크기 차이는 있지만 너무 극단적이지 않게
            prefer_horizontal=1.0,
            random_state=42,
            collocations=False,    # 두 단어를 하나의 프레이즈로 묶지 않기
            # repeat=False (기본값) → 단어는 1번만
        ).generate_from_frequencies(word_scores_dict)

        # 5) 시각화 & 저장
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


    def create_contribution_bar_chart(self, keywords: list, primary_sentiment: str,
                                      output_folder: str, filename: str):
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

        num_charts = len(grouped_keywords)
        if num_charts == 0:
            return

        fig, axes = plt.subplots(num_charts, 1, figsize=(12, 5.5 * num_charts))
        if num_charts == 1:
            axes = [axes]

        fig.suptitle(
            f'문서: {filename} | LLaMA 기반 상황적 키워드 기여도 분석',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )

        max_weight = max(
            item['contribution_weight']
            for item_list in grouped_keywords.values()
            for item in item_list
        ) if grouped_keywords else 0.1

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
                        reason_wrapped = '\n'.join(
                            [reason[i:i + 35] for i in range(0, len(reason), 35)]
                        )
                        text += f'\n({reason_wrapped})'

                    ax.text(
                        bar.get_width() + 0.005,
                        bar.get_y() + bar.get_height() / 2,
                        text,
                        va='center',
                        fontsize=7,
                        ha='left'
                    )

                plot_index += 1

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        output_path = os.path.join(output_folder, f"{filename}_contribution_barchart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ [1] 감성별 기여도 막대 차트 저장: {Path(output_path).name}")

    def create_sentiment_pie_chart(self, primary_sentiment: str, confidence: float,
                                   output_folder: str, filename: str):
        """
        최종 감성과 그 감성에 기여한 키워드별 가중치를 보여주는 원형 차트
        """
        global analysis_data
        keywords = analysis_data.get('contextual_keywords', [])

        # 1. 최종 감성과 일치하는 키워드만 필터링
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
        """단일 파일 시각화 및 결과 폴더 정리"""
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

        # 1. 감성별 기여도 막대 차트
        self.create_contribution_bar_chart(
            keywords, primary_sentiment, output_folder, filename_prefix
        )

        # 2. 최종 감성 키워드 파이 차트
        self.create_sentiment_pie_chart(
            primary_sentiment, confidence, output_folder, filename_prefix
        )

        # 3. 키워드 감성별 워드클라우드 (원형 마스크 적용)
        self.create_wordcloud_chart(keywords, output_folder, filename_prefix)

        # 4. 인터뷰 요약 TXT 생성
        self.create_summary_txt(analysis_data, output_folder)

        print(f"\n   ✅ 시각화 완료! 결과 폴더: {output_folder}")

    def visualize_all_files(self):
        """전체 파일 시각화"""
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

# src/modules/EmoriVisualizer.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

class EmoriVisualizer:
    def __init__(self, output_dir="output/report_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 🎨 컬러 팔레트
        self.colors = {
            'brain_blue': '#4A90E2',   # VR/뇌파 (신체)
            'text_black': '#333333',   # 상담/언어 (현재)
            'safe_green': '#2ECC71',   # 긍정/안정
            'warning_red': '#E74C3C',  # 경고/괴리
            'neutral_gray': '#95A5A6'
        }
        
        self._set_korean_font()

    def _set_korean_font(self):
        """OS별 한글 폰트 자동 설정"""
        font_names = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP"]
        font_found = False
        for name in font_names:
            try:
                if fm.findfont(fm.FontProperties(family=name)):
                    plt.rcParams["font.family"] = name
                    plt.rcParams["axes.unicode_minus"] = False
                    font_found = True
                    break
            except:
                continue
        if not font_found:
            print("⚠️ [Visualizer] 한글 폰트를 찾지 못했습니다. 그래프 글자가 깨질 수 있습니다.")

    def plot_core_radar(self, core_states, filename="radar_core.png"):
        """
        [1페이지] VR(신체) vs 상담(언어) 종합 5각 레이더 차트
        """
        # 데이터 준비 (100점 만점 환산)
        labels = []
        values = []
        for k, v in core_states.items():
            score = int(v * 100)
            labels.append(f"{k}\n({score}점)")
            values.append(score)
        
        # 차트 닫기
        values += [values[0]]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        
        # 1. VR 데이터 영역 (파랑)
        ax.fill(angles, values, color=self.colors['brain_blue'], alpha=0.2)
        ax.plot(angles, values, color=self.colors['brain_blue'], linewidth=2, label='측정 지표')
        
        # 2. 상담 데이터(언어) 축 강조
        for i, label in enumerate(labels):
            if "언어" in label or "상담" in label:
                ax.plot([0, angles[i]], [0, values[i]], color=self.colors['text_black'], linestyle='--', linewidth=1.5)
                ax.scatter(angles[i], values[i], color=self.colors['text_black'], s=100, zorder=10, label='언어/상담 지표')

        # 스타일 설정
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels([]) 
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', pad=20)

        plt.title("VR 생체신호 및 상담 언어 종합 분석", fontsize=15, fontweight='bold', pad=30)
        
        # [UI 개선] 하단 여백 확보하여 글자 겹침 방지
        plt.subplots_adjust(bottom=0.2) # 차트 아래 공간 확보
        
        # 안내 문구를 더 아래로 배치
        plt.figtext(0.5, 0.05, "※ 점수가 높을수록(100점에 가까울수록) 긍정적이고 좋은 상태입니다.", 
                    ha="center", fontsize=10, color="gray", 
                    bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5", ec="none"))
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ [Chart] 레이더 차트 생성 완료: {save_path}")

    def plot_discrepancy(self, stress_score, text_score, filename="discrepancy_bar.png"):
        """
        [2페이지] 신체적 스트레스(VR) vs 언어적 긍정성(상담) 일치 분석
        """
        fig, ax = plt.subplots(figsize=(9, 4))
        
        y_pos = 0
        height = 0.6
        
        # 1. 좌우 막대 그래프
        ax.barh(y_pos, -stress_score, height, align='center', color=self.colors['brain_blue'], alpha=0.8)
        ax.barh(y_pos, text_score, height, align='center', color=self.colors['text_black'], alpha=0.8)
        
        # 중앙선
        ax.axvline(0, color='gray', linewidth=1, linestyle='--')
        
        # 축 설정
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        
        # X축 라벨 (높음/중간/없음)
        ticks_loc = [-1.0, -0.5, 0, 0.5, 1.0]
        ticks_label = ['높음', '중간', '없음', '중간', '높음']
        ax.set_xticks(ticks_loc)
        ax.set_xticklabels(ticks_label, fontsize=10)
        
        # X축 설명
        ax.set_xlabel("(지표 강도)", fontsize=10, color='gray', labelpad=10)
        
        # [UI 개선] 라벨 단순화 (신체/언어)
        ax.text(-0.6, -0.45, "VR 뇌파 스트레스\n(신체)", ha='center', va='top', 
                fontsize=11, fontweight='bold', color=self.colors['brain_blue'])
        ax.text(0.6, -0.45, "상담 대화 긍정성\n(언어)", ha='center', va='top', 
                fontsize=11, fontweight='bold', color=self.colors['text_black'])
        
        # 수치 표시
        ax.text(-stress_score - 0.05, y_pos, f"{stress_score:.2f}", va='center', ha='right', fontweight='bold', color='black')
        ax.text(text_score + 0.05, y_pos, f"{text_score:.2f}", va='center', ha='left', fontweight='bold', color='black')

        # [UI 개선] 이모티콘 제거 및 제목 변경
        if stress_score > 0.5 and text_score > 0.5:
            # 경고 상태
            ax.text(0, 0.55, "잠재적 우울/괴리 의심 (몸은 힘들지만 말은 긍정적임)", 
                    ha='center', va='center', fontsize=11, color=self.colors['warning_red'], fontweight='bold',
                    bbox=dict(facecolor='#FFF0F0', edgecolor=self.colors['warning_red'], boxstyle='round,pad=0.4'))
        else:
            # 정상 상태
            ax.text(0, 0.55, "신체 반응과 언어 표현이 비교적 일치합니다.", 
                    ha='center', va='center', fontsize=10, color='gray')

        # 제목 변경: 불일치 -> 일치
        plt.title("신체(VR) - 언어(상담) 일치 분석", fontsize=15, fontweight='bold', pad=20)
        
        plt.subplots_adjust(bottom=0.25)
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ [Chart] 일치 분석 차트 생성 완료: {save_path}")

    def plot_vr_flow(self, steps, values, keywords, filename="flow_chart.png"):
        """
        [2페이지] VR 단계별 감정 흐름
        """
        custom_steps = ['VR 체험 전\n(안정)', 'VR 체험 중\n(활동)', 'VR 종료 후\n(결과)']
        
        if len(steps) == len(custom_steps):
            display_steps = custom_steps
        else:
            display_steps = steps

        x = range(len(display_steps))
        
        fig, ax = plt.subplots(figsize=(9, 4.5))
        
        ax.plot(x, values, marker='o', linewidth=2.5, markersize=9, color=self.colors['safe_green'])
        ax.fill_between(x, values, alpha=0.1, color=self.colors['safe_green'])
        
        if keywords and isinstance(keywords, list):
            keyword_text = ", ".join(keywords[:3])
        else:
            keyword_text = "주요 토픽 없음"

        last_x = x[-1]
        last_y = values[-1]
        
        ax.axvline(last_x, color='gray', linestyle=':', alpha=0.5)
        
        ax.annotate(f"이어진 상담 주요 토픽:\n'{keyword_text}'", 
                    xy=(last_x, last_y), xytext=(last_x - 0.2, last_y + 0.2),
                    bbox=dict(boxstyle="round,pad=0.6", fc="#FFFFE0", ec="orange", alpha=1.0, linestyle='--'),
                    fontsize=10, fontweight='bold', ha='right',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='gray'))

        ax.set_xticks(x)
        ax.set_xticklabels(display_steps, fontsize=11)
        ax.set_ylim(0, 1.2) 
        ax.set_ylabel("VR 감정 에너지 (활성도)", fontsize=10)
        
        plt.title("VR 체험 감정 변화 및 상담 토픽", fontsize=15, fontweight='bold', pad=15)
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ [Chart] 흐름 차트 생성 완료: {save_path}")
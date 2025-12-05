import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os
import json
from pathlib import Path


current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from EmoriAnalyzer import EmoriAnalyzer
except ImportError:
    
    try:
        from src.modules.EmoriAnalyzer import EmoriAnalyzer
    except ImportError:
        print("EmoriAnalyzer를 찾을 수 없습니다. 연동 테스트가 불가능합니다.")
        EmoriAnalyzer = None

class EmoriVisualizer:
    def __init__(self, output_dir="output/report_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.colors = {
            'brain_blue': "#ED1212",   
            'text_black': '#333333',   
            'warning_red': "#6F3CE7",  
            'safe_gray': '#7F8C8D'     
        }
        
        self._set_korean_font()

    def _set_korean_font(self):
                             
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
            print("한글 폰트를 찾지 못했습니다.")

    def plot_discrepancy(self, stress_score, text_score, filename="discrepancy_bar.png"):
        fig, ax = plt.subplots(figsize=(9, 4))
        
        y_pos = 0
        height = 0.6
        
        
        ax.barh(y_pos, -stress_score, height, align='center', color=self.colors['brain_blue'], alpha=0.9)
        ax.barh(y_pos, text_score, height, align='center', color=self.colors['text_black'], alpha=0.9)
    
        ax.axvline(0, color='gray', linewidth=1.5, linestyle='-')
        
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        
        ticks_loc = [-1.0, -0.5, 0, 0.5, 1.0]
        ticks_label = ['높음', '중간', '없음', '중간', '높음']
        ax.set_xticks(ticks_loc)
        ax.set_xticklabels(ticks_label, fontsize=10, fontweight='bold', color='#555555')
        ax.set_xlabel("(지표 강도)", fontsize=10, color='gray', labelpad=8)
        
        ax.text(-0.6, -0.45, "VR 뇌파로 나타나는 스트레스\n(신체)", ha='center', va='top', 
                fontsize=12, fontweight='bold', color=self.colors['brain_blue'])
        ax.text(0.6, -0.45, "상담 대화로 나타나는 긍정성\n(언어)", ha='center', va='top', 
                fontsize=12, fontweight='bold', color=self.colors['text_black'])
        
        ax.text(-stress_score - 0.05, y_pos, f"{stress_score:.2f}", va='center', ha='right', fontweight='bold', fontsize=11)
        ax.text(text_score + 0.05, y_pos, f"{text_score:.2f}", va='center', ha='left', fontweight='bold', fontsize=11)

        if stress_score > 0.6 and text_score > 0.6:
            ax.text(0, 0.55, "잠재적 우울/괴리 의심 (몸은 힘들지만 말은 긍정적임)", 
                    ha='center', va='center', fontsize=11, color=self.colors['warning_red'], fontweight='bold',
                    bbox=dict(facecolor='#FFF0F0', edgecolor=self.colors['warning_red'], boxstyle='round,pad=0.4'))
        else:
            ax.text(0, 0.55, "신체 반응과 언어 표현이 비교적 일치합니다.", 
                    ha='center', va='center', fontsize=11, color=self.colors['safe_gray'], fontweight='bold')

        plt.title("신체(VR) - 언어(상담) 일치 분석", fontsize=16, fontweight='bold', pad=30)
        
        plt.subplots_adjust(bottom=0.25)
        
        save_path = self.output_dir / filename
        
        # 파일이 이미 존재하면 스킵
        if save_path.exists():
            print(f"⚠️ 일치 분석 차트 파일이 이미 존재합니다. 스킵: {save_path}")
            plt.close()
            return
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"차트 생성 완료: {save_path}")


if __name__ == "__main__":
    if EmoriAnalyzer is None:
        print("❌ Analyzer 모듈이 없어 연동 테스트를 중단합니다.")
        sys.exit()

    print("\n🚀 [Integration Test] Analyzer -> Visualizer 데이터 연결 테스트 시작...")

    
    base_dir = Path(os.getcwd())
    
    
    eeg_path = base_dir / "output/Emotion_EEG/Report_Json_Data/Report_Data.json"
    llama_path = base_dir / "output/llama3/EB_001_llama_analysis.json"
    
    print(f"📂 EEG 데이터 로드: {eeg_path}")
    print(f"📂 감정 데이터 로드: {llama_path}")

    if eeg_path.exists() and llama_path.exists():
        analyzer = EmoriAnalyzer(str(eeg_path), str(llama_path))
        result = analyzer.analyze("participant_1")
        
        if result:
            real_stress = result['discrepancy']['stress_val']
            real_text = result['discrepancy']['text_val']
            
            print(f"\n Analyzer 결과 수신")
            print(f"   - 실제 스트레스 수치: {real_stress:.2f}")
            print(f"   - 실제 언어 긍정 수치: {real_text:.2f}")
            
            print("\n [Visualizer] 그래프 생성 중...")
            viz = EmoriVisualizer(output_dir="output/report_images")
            
            viz.plot_discrepancy(
                stress_score=real_stress, 
                text_score=real_text, 
                filename="real_data_discrepancy.png"
            )
            
            print(f"\n 연동 성공! 'output/report_images/real_data_discrepancy.png' 파일을 확인하세요.")
        else:
            print(" 분석 결과가 없습니다.")
    else:
        print(" 테스트에 필요한 데이터 파일이 없습니다. 경로를 확인해주세요.")
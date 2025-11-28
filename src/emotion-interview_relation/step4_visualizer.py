import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pathlib import Path
import numpy as np
import platform

# --- 설정 ---
FINAL_INPUT_DIR = 'output/emotionRelation/finalRelation'
VISUAL_OUTPUT_DIR = 'output/emotionRelation/visualization'
os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)

BLUE_TO_BLUE_THRESHOLD = 0.7 

# ✅ 감정 한글 매핑 (일단은 컬러 결정용으로만 사용)
EMOTION_TRANSLATION = {
    # 기쁘다 계열 (노란색)
    "Happy": "기쁘다",
    # 슬프다 계열 (파란색)
    "Sad": "슬프다",
    # 화나다 계열 (빨간색)
    "Angry": "화나다",
    # 두렵다 계열 (초록색)
    "fear": "두렵다",
    # 놀라다 계열 (하늘색)
    "Surprise": "놀라다",
    # 싫다 계열 (보라색)
    "Dislike": "싫다",
}

# ✅ 메인 감정 노드 색상 매핑
EMOTION_COLOR = {
    "Happy": "#f1c40f",     # 노란색
    "Sad": "#2980b9",       # 파란색
    "Angry": "#e74c3c",     # 빨강
    "fear": "#27ae60",      # 초록
    "Surprise": "#5dade2",  # 하늘색
    "Dislike": "#9b59b6",   # 보라
}


class NetworkVisualizerPNG:
    def __init__(self):
        print("✅ 네트워크 시각화 준비 완료")
        self.font_prop = self._setup_font()
        
        if self.font_prop:
            rc('font', family=self.font_prop.get_name())
            plt.rcParams['axes.unicode_minus'] = False

    def _setup_font(self):
        """운영체제에 맞는 한글 폰트 자동 설정"""
        system = platform.system()
        font_candidates = []
        
        font_dir = 'fonts'
        if os.path.exists(font_dir):
            for font_file in ['AppleGothic.ttf', 'malgun.ttf', 'NanumGothic.ttf']:
                font_path = os.path.join(font_dir, font_file)
                if os.path.exists(font_path):
                    font_candidates.append(font_path)
        
        if system == 'Darwin':
            font_candidates.extend([
                '/System/Library/Fonts/AppleGothic.ttf',
                '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
                '/Library/Fonts/AppleGothic.ttf'
            ])
        elif system == 'Windows':
            font_candidates.extend([
                'C:/Windows/Fonts/malgun.ttf',
                'C:/Windows/Fonts/gulim.ttc'
            ])
        elif system == 'Linux':
            font_candidates.extend([
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
            ])
        
        for font_path in font_candidates:
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                print(f"✅ 폰트 설정: {font_prop.get_name()}")
                return font_prop
        
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        korean_fonts = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Noto Sans KR']
        
        for korean_font in korean_fonts:
            if korean_font in available_fonts:
                font_prop = font_manager.FontProperties(family=korean_font)
                print(f"✅ 시스템 폰트 사용: {korean_font}")
                return font_prop
        
        print("⚠️ 한글 폰트를 찾을 수 없습니다.")
        return font_manager.FontProperties()

    def load_final_data(self, filename: str) -> dict | None:
        file_path = os.path.join(FINAL_INPUT_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"🛑 파일 없음: {file_path}")
            return None

    def create_weighted_circular_layout(self, main_node, connected_nodes, edge_weights):
        """가중치 기반 원형 레이아웃 - 가중치 높을수록 중심에 가까이"""
        pos = {}
        
        # 메인 노드는 정확히 중앙
        pos[main_node] = np.array([0.0, 0.0])
        
        if not connected_nodes:
            return pos
        
        n = len(connected_nodes)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        for i, node in enumerate(connected_nodes):
            weight = edge_weights.get((main_node, node), 0.3)
            
            if weight >= 0.7:
                distance = 1.5
            elif weight >= 0.5:
                distance = 2.5
            elif weight >= 0.3:
                distance = 4.0
            else:
                distance = 5.5
            
            x = distance * np.cos(angles[i])
            y = distance * np.sin(angles[i])
            pos[node] = np.array([x, y])
        
        return pos

    def visualize_single_file(self, filename: str):
        data = self.load_final_data(filename)
        if not data:
            return
        
        base_name = Path(filename).stem.removesuffix('_finalRelation')
        
        main_emotion_word = data['main_emotion_node']['word']
        blue_dot_keywords_data = {k['word']: k for k in data['keyword_nodes']}

        # ✅ 메인 감정에 따른 색상 선택
        main_emotion_color = EMOTION_COLOR.get(main_emotion_word, '#2c3e50')

        # is_connected 기준으로 분리
        connected_to_main = []
        disconnected_from_main = []

        for relation in data['inter_node_relations']:
            word = relation['target']
            if relation['is_connected']:
                connected_to_main.append(word)
            else:
                disconnected_from_main.append(word)

        print(f"\n📊 {base_name} 분석:")
        print(f"   ✅ 연결된 키워드: {len(connected_to_main)}개 - {connected_to_main}")
        print(f"   ❌ 단절된 키워드: {len(disconnected_from_main)}개 - {disconnected_from_main}")

        # ===== 그래프 1: 연결 그룹 =====
        if connected_to_main:
            fig, ax = plt.subplots(figsize=(24, 24), dpi=300)
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')
            
            G_connected = nx.Graph()
            G_connected.add_node(main_emotion_word)
            for word in connected_to_main:
                G_connected.add_node(word)
            
            edge_weights = {}
            
            # Black-Blue 연결 (메인 감정 ↔ 키워드)
            for relation in data['inter_node_relations']:
                if relation['target'] in connected_to_main and relation['is_connected']:
                    edge = (main_emotion_word, relation['target'])
                    edge_weights[edge] = relation['sbert_score']
                    G_connected.add_edge(*edge, weight=relation['sbert_score'])
            
            # Blue-Blue 연결
            blue_edges = []
            if 'intra_node_relations' in data:
                for relation in data['intra_node_relations']:
                    if (relation['word1'] in connected_to_main and 
                        relation['word2'] in connected_to_main and 
                        relation['sbert_score'] >= BLUE_TO_BLUE_THRESHOLD):
                        edge = (relation['word1'], relation['word2'])
                        blue_edges.append((edge, relation['sbert_score']))
                        G_connected.add_edge(*edge, weight=relation['sbert_score'])
            
            # 가중치 기반 원형 레이아웃
            pos = self.create_weighted_circular_layout(
                main_emotion_word, connected_to_main, edge_weights
            )
            
            # ✅ 노드 크기 크게 조정
            node_sizes = []
            node_colors = []
            
            for node in G_connected.nodes():
                if node == main_emotion_word:
                    node_sizes.append(14000)  # 메인 노드 더 크게
                    node_colors.append(main_emotion_color)
                else:
                    keyword_data = blue_dot_keywords_data[node]
                    node_sizes.append(6000 + keyword_data['contribution_weight'] * 4000)
                    node_colors.append('#3498db')
            
            # ✅ Black-Blue 엣지: 두께/진하기 강화
            for edge, weight in edge_weights.items():
                if weight >= 0.7:
                    color = '#e74c3c'  # 강한 연결
                    width = 12
                    alpha = 1.0
                elif weight >= 0.5:
                    color = '#e67e22'
                    width = 8
                    alpha = 0.85
                elif weight >= 0.3:
                    color = '#f39c12'
                    width = 5
                    alpha = 0.65
                else:
                    color = '#95a5a6'
                    width = 3
                    alpha = 0.4
                
                nx.draw_networkx_edges(
                    G_connected, pos,
                    edgelist=[edge],
                    edge_color=[color],
                    width=width,
                    alpha=alpha,
                    style='solid',
                    ax=ax
                )
            
            # ✅ Blue-Blue 엣지도 더 두껍게
            for edge, weight in blue_edges:
                nx.draw_networkx_edges(
                    G_connected, pos,
                    edgelist=[edge],
                    edge_color=['#3498db'],
                    width=3 + weight * 3,   # (대략 3~6)
                    alpha=0.5,
                    style='dashed',
                    ax=ax
                )
            
            # 노드 그리기
            nx.draw_networkx_nodes(
                G_connected, pos,
                node_color=node_colors,
                node_size=node_sizes,
                edgecolors='white',
                linewidths=4,
                ax=ax
            )
            
            # 라벨 그리기
            for node in G_connected.nodes():
                x, y = pos[node]
                
                if node == main_emotion_word:
                    bbox_props = dict(
                        boxstyle='round,pad=0.8',
                        facecolor=main_emotion_color,   # ✅ 감정 색상 사용
                        edgecolor='white',
                        linewidth=3,
                        alpha=0.97
                    )
                    ax.text(
                        x, y, node,
                        fontsize=34,            # ✅ 글씨 키움
                        fontweight='bold',
                        color='white',
                        ha='center',
                        va='center',
                        bbox=bbox_props,
                        fontproperties=self.font_prop,
                        zorder=1000
                    )
                else:
                    keyword_data = blue_dot_keywords_data[node]
                    sentiment = keyword_data['sentiment_label'][:1]
                    
                    bbox_props = dict(
                        boxstyle='round,pad=0.6',
                        facecolor='white',
                        edgecolor='#3498db',
                        linewidth=2.5,
                        alpha=0.97
                    )
                    ax.text(
                        x, y, f"{node}\n({sentiment})",
                        fontsize=20,        # ✅ 글씨 키움
                        fontweight='bold',
                        color='black',
                        ha='center',
                        va='center',
                        bbox=bbox_props,
                        fontproperties=self.font_prop,
                        zorder=1000
                    )
            
            ax.set_title(
                f"심리 연관성 네트워크: {base_name} (연결 그룹)",
                fontsize=30,
                fontweight='bold',
                pad=40,
                fontproperties=self.font_prop
            )
            ax.axis('off')
            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            
            plt.tight_layout()
            output_path = os.path.join(VISUAL_OUTPUT_DIR, f"{base_name}_connected_group.png")
            plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300, facecolor='#f8f9fa')
            plt.close()
            print(f"  ✅ 저장: {output_path}")
        
        else:
            print(f"  ℹ️ 연결된 키워드가 없음")

        # ===== 그래프 2: 단절 그룹 =====
        if disconnected_from_main:
            fig, ax = plt.subplots(figsize=(22, 22), dpi=300)
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')
            
            G_disconnected = nx.Graph()
            
            for word in disconnected_from_main:
                G_disconnected.add_node(word)
            
            edge_list = []
            
            if 'intra_node_relations' in data:
                for relation in data['intra_node_relations']:
                    if (relation['word1'] in disconnected_from_main and 
                        relation['word2'] in disconnected_from_main and 
                        relation['sbert_score'] >= BLUE_TO_BLUE_THRESHOLD):
                        edge = (relation['word1'], relation['word2'])
                        edge_list.append((edge, relation['sbert_score']))
                        G_disconnected.add_edge(*edge, weight=relation['sbert_score'])
            
            # spring 레이아웃
            pos = nx.spring_layout(
                G_disconnected, 
                k=2.0,
                iterations=150, 
                seed=42,
                center=(0, 0)
            )
            
            # ✅ 단절 그룹도 노드 크게
            node_sizes = []
            node_colors = []
            
            for node in G_disconnected.nodes():
                keyword_data = blue_dot_keywords_data[node]
                node_sizes.append(5000 + keyword_data['contribution_weight'] * 3500)
                node_colors.append('#95a5a6')
            
            # ✅ 엣지 두께/진하기 강화
            for edge, weight in edge_list:
                nx.draw_networkx_edges(
                    G_disconnected, pos,
                    edgelist=[edge],
                    edge_color='#bdc3c7',
                    width=3 + weight * 4,   # (대략 3~7)
                    alpha=0.6,
                    style='dashed',
                    ax=ax
                )
            
            nx.draw_networkx_nodes(
                G_disconnected, pos,
                node_color=node_colors,
                node_size=node_sizes,
                edgecolors='white',
                linewidths=3,
                ax=ax
            )
            
            # 라벨
            for node in G_disconnected.nodes():
                x, y = pos[node]
                keyword_data = blue_dot_keywords_data[node]
                sentiment = keyword_data['sentiment_label'][:1]
                
                ax.text(
                    x, y, f"{node}\n({sentiment})",
                    fontsize=18,        # ✅ 글씨 키움
                    fontweight='bold',
                    color='black',
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        facecolor='white',
                        edgecolor='#95a5a6',
                        linewidth=2,
                        alpha=0.97
                    ),
                    fontproperties=self.font_prop,
                    zorder=1000
                )
            
            ax.set_title(
                f"심리 연관성 네트워크: {base_name} (단절 그룹)",
                fontsize=30,
                fontweight='bold',
                pad=40,
                fontproperties=self.font_prop
            )
            ax.axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(VISUAL_OUTPUT_DIR, f"{base_name}_disconnected_group.png")
            plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300, facecolor='#f8f9fa')
            plt.close()
            print(f"  ✅ 저장: {output_path}")
        
        else:
            print(f"  ℹ️ 단절된 키워드가 없음")


def main():
    os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)
    
    try:
        visualizer = NetworkVisualizerPNG()
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    print("\n" + "="*60)
    print("🎯 네트워크 맵 시각화 (완성판)")
    print("="*60)
    
    choice = input("\n선택 (1: 단일 파일, 2: 전체 파일, 3: 종료): ")
    
    if choice == '1':
        filename = input("파일명 입력 (예: 김시원_finalRelation.json): ")
        if filename:
            visualizer.visualize_single_file(filename)
        else:
            print("🛑 파일명이 입력되지 않았습니다.")
            
    elif choice == '2':
        print("💡 현재 '전체 파일 처리'는 지원하지 않습니다.")
    
    elif choice == '3':
        print("프로그램을 종료합니다.")
    
    else:
        print("잘못된 입력입니다.")

if __name__ == "__main__":
    main()

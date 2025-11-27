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
SIMILARITY_THRESHOLD = 0.45 

class NetworkVisualizerPNG:
    def __init__(self):
        print("✅ 네트워크 시각화 준비 완료")
        self.font_prop = self._setup_font()
        
        if self.font_prop:
            rc('font', family=self.font_prop.get_name())
            # 마이너스 기호 깨짐 방지
            plt.rcParams['axes.unicode_minus'] = False

    def _setup_font(self):
        """운영체제에 맞는 한글 폰트 자동 설정"""
        system = platform.system()
        font_candidates = []
        
        # 1. fonts 폴더의 폰트 파일 확인
        font_dir = 'fonts'
        if os.path.exists(font_dir):
            for font_file in ['AppleGothic.ttf', 'malgun.ttf', 'NanumGothic.ttf']:
                font_path = os.path.join(font_dir, font_file)
                if os.path.exists(font_path):
                    font_candidates.append(font_path)
        
        # 2. 시스템 폰트 경로
        if system == 'Darwin':  # macOS
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
        
        # 3. 첫 번째로 찾은 폰트 사용
        for font_path in font_candidates:
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                print(f"✅ 폰트 설정 완료: {font_prop.get_name()} ({font_path})")
                return font_prop
        
        # 4. 시스템에 설치된 한글 폰트 검색
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        korean_fonts = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Noto Sans KR']
        
        for korean_font in korean_fonts:
            if korean_font in available_fonts:
                font_prop = font_manager.FontProperties(family=korean_font)
                print(f"✅ 시스템 폰트 사용: {korean_font}")
                return font_prop
        
        print("⚠️ 한글 폰트를 찾을 수 없습니다. 글자가 깨질 수 있습니다.")
        print("💡 해결 방법:")
        print("   - macOS: 기본 설치되어 있음 (재부팅 필요할 수 있음)")
        print("   - Windows: 맑은 고딕이 설치되어 있어야 함")
        print("   - Linux: sudo apt-get install fonts-nanum")
        return font_manager.FontProperties()

    def load_final_data(self, filename: str) -> dict | None:
        file_path = os.path.join(FINAL_INPUT_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"🛑 파일 없음: {file_path}")
            return None

    def create_and_save_graph(self, G: nx.Graph, pos: dict, node_sizes: dict, 
                            node_colors: dict, node_labels: dict, edge_list: list, 
                            edge_colors: list, edge_widths: list, 
                            title: str, output_suffix: str, base_name: str):
        """그래프를 그리고 PNG 파일로 저장"""
        
        plt.figure(figsize=(16, 16), dpi=150)
        
        # 노드 그리기
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=[node_colors.get(n, 'gray') for n in G.nodes()], 
            node_size=[node_sizes.get(n, 500) for n in G.nodes()], 
            edgecolors='black', 
            linewidths=2
        )
        
        # 엣지 그리기
        if edge_list:
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=edge_list, 
                edge_color=edge_colors, 
                width=edge_widths,
                alpha=0.7
            )
        
        # 라벨 그리기
        nx.draw_networkx_labels(
            G, pos, 
            labels=node_labels, 
            font_size=11, 
            font_weight='bold', 
            font_color='black',
            font_family=self.font_prop.get_name() if self.font_prop else None
        )
        
        plt.title(title, fontsize=20, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()

        output_path = os.path.join(VISUAL_OUTPUT_DIR, f"{base_name}_{output_suffix}.png")
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✅ 저장 완료: {output_path}")

    def visualize_single_file(self, filename: str):
        data = self.load_final_data(filename)
        if not data:
            return
        
        base_name = Path(filename).stem.removesuffix('_finalRelation')
        
        main_emotion_word = data['main_emotion_node']['word']
        blue_dot_keywords_data = {k['word']: k for k in data['keyword_nodes']}

        # 연결 상태 분류
        connected_to_main = []
        disconnected_from_main = []

        for relation in data['inter_node_relations']:
            word = relation['target']
            if relation['is_connected']:
                connected_to_main.append(word)
            else:
                disconnected_from_main.append(word)

        print(f"\n📊 {base_name} 분석 결과:")
        print(f"   - Black Dot과 연결된 키워드: {len(connected_to_main)}개")
        print(f"   - Black Dot과 단절된 키워드: {len(disconnected_from_main)}개")

        # ===== 그래프 1: 연결 그룹 =====
        G_connected = nx.Graph()
        node_sizes_conn = {}
        node_colors_conn = {}
        node_labels_conn = {}
        edge_list_conn = []
        edge_colors_conn = []
        edge_widths_conn = []

        G_connected.add_node(main_emotion_word)
        node_sizes_conn[main_emotion_word] = 2000
        node_colors_conn[main_emotion_word] = '#000000'
        node_labels_conn[main_emotion_word] = main_emotion_word

        for word in connected_to_main:
            keyword_data = blue_dot_keywords_data[word]
            G_connected.add_node(word)
            node_sizes_conn[word] = 1000 + (keyword_data['contribution_weight'] * 800)
            node_colors_conn[word] = '#42A5F5'
            sentiment = keyword_data['sentiment_label']
            node_labels_conn[word] = f"{word}\n({sentiment[:1]})"

            # Black Dot과의 연결
            for relation in data['inter_node_relations']:
                if relation['source'] == main_emotion_word and relation['target'] == word:
                    if relation['is_connected']:
                        edge_list_conn.append((relation['source'], relation['target']))
                        edge_colors_conn.append('#333333')
                        edge_widths_conn.append(max(2, relation['sbert_score'] * 5))

        # Blue Dot 간 연결
        if 'intra_node_relations' in data:
            for relation in data['intra_node_relations']:
                if (relation['word1'] in connected_to_main and 
                    relation['word2'] in connected_to_main and 
                    relation['sbert_score'] >= BLUE_TO_BLUE_THRESHOLD):
                    edge_list_conn.append((relation['word1'], relation['word2']))
                    edge_colors_conn.append('#90CAF9')
                    edge_widths_conn.append(max(1, relation['sbert_score'] * 2))

        # 레이아웃
        pos_conn = nx.spring_layout(G_connected, k=1.0, iterations=100, seed=42)
        pos_conn[main_emotion_word] = np.array([0, 0])

        self.create_and_save_graph(
            G_connected, pos_conn, node_sizes_conn, node_colors_conn, 
            node_labels_conn, edge_list_conn, edge_colors_conn, edge_widths_conn, 
            f"심리 연관성 네트워크: {base_name} (연결 그룹)", 
            "connected_group", base_name
        )

        # ===== 그래프 2: 단절 그룹 =====
        if disconnected_from_main:
            G_disconnected = nx.Graph()
            node_sizes_disconn = {}
            node_colors_disconn = {}
            node_labels_disconn = {}
            edge_list_disconn = []
            edge_colors_disconn = []
            edge_widths_disconn = []

            for word in disconnected_from_main:
                keyword_data = blue_dot_keywords_data[word]
                G_disconnected.add_node(word)
                
                node_sizes_disconn[word] = 1000 + (keyword_data['contribution_weight'] * 800)
                node_colors_disconn[word] = '#42A5F5'
                sentiment = keyword_data['sentiment_label']
                node_labels_disconn[word] = f"{word}\n({sentiment[:1]})"
            
            # Blue Dot 간 연결
            if 'intra_node_relations' in data:
                for relation in data['intra_node_relations']:
                    if (relation['word1'] in disconnected_from_main and 
                        relation['word2'] in disconnected_from_main and 
                        relation['sbert_score'] >= BLUE_TO_BLUE_THRESHOLD):
                        edge_list_disconn.append((relation['word1'], relation['word2']))
                        edge_colors_disconn.append('#90CAF9')
                        edge_widths_disconn.append(max(1, relation['sbert_score'] * 2))
            
            pos_disconn = nx.spring_layout(G_disconnected, k=1.0, iterations=100, seed=42)

            self.create_and_save_graph(
                G_disconnected, pos_disconn, node_sizes_disconn, node_colors_disconn, 
                node_labels_disconn, edge_list_disconn, edge_colors_disconn, edge_widths_disconn, 
                f"심리 연관성 네트워크: {base_name} (단절 그룹)", 
                "disconnected_group", base_name
            )
        else:
            print(f"  ℹ️ 단절된 키워드가 없어 단절 그룹 그래프는 생성되지 않습니다.")

def main():
    os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)
    
    try:
        visualizer = NetworkVisualizerPNG()
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    print("\n" + "="*50)
    print("🎯 네트워크 맵 시각화 (PNG)")
    print("="*50)
    
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
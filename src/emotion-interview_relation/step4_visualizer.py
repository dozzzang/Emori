import os
import json
import networkx as nx
from pathlib import Path
from pyvis.network import Network


FINAL_INPUT_DIR = 'output/emotionRelation/finalRelation'
VISUAL_OUTPUT_DIR = 'output/emotionRelation/visualization'
os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)

BLUE_TO_BLUE_THRESHOLD = 0.7 


EMOTION_COLOR = {
    "Happy": "#f1c40f",     
    "Sad": "#2980b9",       
    "Angry": "#e74c3c",      
    "fear": "#27ae60",       
    "Surprise": "#5dade2",  
    "Dislike": "#9b59b6",   
}


PHYSICS_OPTIONS = {
    "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -80,
            "centralGravity": 0.02,
            "springLength": 150,
            "springConstant": 0.08,
            "avoidOverlap": 1
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "stabilization": {
            "enabled": True,
            "iterations": 1000,
            "updateInterval": 25,
            "onlyDynamicEdges": False,
            "fit": True
        }
    }
}


CUSTOM_CSS = """
<style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
    
    body { 
        font-family: 'Pretendard', sans-serif !important; 
        background-color: #ffffff !important;
        margin: 0; padding: 0; overflow: hidden;
    }
    
    div.vis-tooltip {
        background-color: rgba(255, 255, 255, 0.98) !important;
        color: #333 !important;
        border: 1px solid #ddd !important;
        border-radius: 12px !important;
        padding: 15px !important;
        font-size: 15px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important;
    }
</style>
"""


CUSTOM_JS = """
<script>
    network.on("stabilizationIterationsDone", function () {
        console.log("배치 완료. 줌을 맞춥니다.");
        network.fit({ 
            animation: { duration: 1000, easingFunction: "easeInOutQuad" } 
        });
        
        setTimeout(function() {
            console.log("움직임 정지 (Freeze)");
            network.setOptions({ physics: { enabled: false } });
        }, 1200);
    });
    
    setTimeout(function() {
        network.fit();
        network.setOptions({ physics: { enabled: false } });
    }, 2000);
</script>
"""


class NetworkVisualizerHTML:
    def __init__(self):
        print("✅ 네트워크 시각화 준비 완료 (HTML)")

    def load_final_data(self, filename: str) -> dict | None:
        file_path = os.path.join(FINAL_INPUT_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"🛑 파일 없음: {file_path}")
            return None

    def visualize_single_file(self, filename: str):
        data = self.load_final_data(filename)
        if not data:
            return
        
        base_name = Path(filename).stem.removesuffix('_finalRelation')
        
        main_emotion_word = data['main_emotion_node']['word']
        blue_dot_keywords_data = {k['word']: k for k in data['keyword_nodes']}

        
        main_emotion_color = EMOTION_COLOR.get(main_emotion_word, '#2c3e50')

        
        connected_to_main = []
        disconnected_from_main = []

        for relation in data['inter_node_relations']:
            word = relation['target']
            if relation['is_connected']:
                connected_to_main.append(word)
            else:
                disconnected_from_main.append(word)

        print(f"\n📊 {base_name} 분석:")
        print(f"   ✅ 연결된 키워드: {len(connected_to_main)}개")
        print(f"   ❌ 단절된 키워드: {len(disconnected_from_main)}개")

        
        if connected_to_main:
            self._create_connected_graph(
                base_name, main_emotion_word, connected_to_main, 
                blue_dot_keywords_data, data, main_emotion_color
            )
        else:
            print(f"  ℹ️ 연결된 키워드가 없음")

        
        if disconnected_from_main:
            self._create_disconnected_graph(
                base_name, disconnected_from_main, 
                blue_dot_keywords_data, data
            )
        else:
            print(f"  ℹ️ 단절된 키워드가 없음")
            empty_html_path = os.path.join(VISUAL_OUTPUT_DIR, f"{base_name}_disconnected_group.html")
            with open(empty_html_path, 'w', encoding='utf-8') as f:
                f.write("""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>단절된 키워드 그룹 없음</title>
    <style>
        body { font-family: 'Pretendard', sans-serif; text-align: center; padding: 50px; }
    </style>
</head>
<body>
    <h2>단절된 키워드 그룹이 없습니다.</h2>
</body>
</html>""")
            print(f"  ✅ 빈 HTML 파일 생성: {empty_html_path}")

    def _create_connected_graph(self, base_name, main_emotion_word, connected_nodes, 
                               blue_dot_keywords_data, data, main_emotion_color):
        
        net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.set_options(json.dumps(PHYSICS_OPTIONS))

        
        G_connected = nx.Graph()
        G_connected.add_node(main_emotion_word)
        for word in connected_nodes:
            G_connected.add_node(word)
        
        edge_weights = {}
        
        for relation in data['inter_node_relations']:
            if relation['target'] in connected_nodes and relation['is_connected']:
                edge = (main_emotion_word, relation['target'])
                edge_weights[edge] = relation['sbert_score']
                G_connected.add_edge(*edge, weight=relation['sbert_score'])
        
        blue_edges = []
        if 'intra_node_relations' in data:
            for relation in data['intra_node_relations']:
                if (relation['word1'] in connected_nodes and 
                    relation['word2'] in connected_nodes and 
                    relation['sbert_score'] >= BLUE_TO_BLUE_THRESHOLD):
                    edge = (relation['word1'], relation['word2'])
                    blue_edges.append((edge, relation['sbert_score']))
                    G_connected.add_edge(*edge, weight=relation['sbert_score'])

        
        net.add_node(
            n_id=main_emotion_word,
            label=main_emotion_word,
            title=f"메인 감정: {main_emotion_word}",
            color=main_emotion_color,
            shape="dot",
            value=50,
            font={'size': 30, 'face': 'Pretendard', 'color': '#ffffff', 'bold': True},
            borderWidth=0,
            shadow={'enabled': True, 'color': 'rgba(0,0,0,0.3)', 'size': 10}
        )

        
        for word in connected_nodes:
            keyword_data = blue_dot_keywords_data[word]
            sentiment = keyword_data['sentiment_label']
            weight = keyword_data['contribution_weight']
            
            size = 20 + (weight * 30)
            
            tooltip = f"키워드: {word}\n감성: {sentiment}\n기여도: {weight:.3f}"
            if 'reason' in keyword_data:
                tooltip += f"\n근거: {keyword_data['reason']}"
            
            net.add_node(
                n_id=word,
                label=word,
                title=tooltip,
                color="#3498db",
                shape="dot",
                value=size,
                font={'size': 18, 'face': 'Pretendard', 'color': '#333333', 'bold': True},
                borderWidth=2,
                borderColor="#ffffff",
                shadow={'enabled': True, 'color': 'rgba(0,0,0,0.2)', 'size': 5}
            )
            
            if (main_emotion_word, word) in edge_weights:
                weight_val = edge_weights[(main_emotion_word, word)]
                edge_color = self._get_edge_color(weight_val)
                edge_width = self._get_edge_width(weight_val)
                
                net.add_edge(
                    main_emotion_word, word,
                    width=edge_width,
                    color=edge_color,
                    title=f"연관도: {weight_val:.3f}"
                )

        
        for edge, weight in blue_edges:
            net.add_edge(
                edge[0], edge[1],
                width=weight * 5,
                color="#3498db",
                dashes=True,
                title=f"키워드 간 연관도: {weight:.3f}"
            )

        
        output_path = os.path.join(VISUAL_OUTPUT_DIR, f"{base_name}_connected_group.html")
        net.save_graph(output_path)
        self._inject_custom_scripts(output_path, f"연결된 키워드 그룹: {base_name}")
        print(f"  ✅ 저장: {output_path}")

    def _create_disconnected_graph(self, base_name, disconnected_nodes, 
                                   blue_dot_keywords_data, data):
        
        net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="#333333")
        
        physics_options = PHYSICS_OPTIONS.copy()
        if len(disconnected_nodes) <= 2:
            physics_options["physics"]["forceAtlas2Based"]["gravitationalConstant"] = -50
            physics_options["physics"]["forceAtlas2Based"]["springLength"] = 200
        
        net.set_options(json.dumps(physics_options))

        G_disconnected = nx.Graph()
        for word in disconnected_nodes:
            G_disconnected.add_node(word)

        edge_list = []
        if 'intra_node_relations' in data:
            for relation in data['intra_node_relations']:
                if (relation['word1'] in disconnected_nodes and 
                    relation['word2'] in disconnected_nodes and 
                    relation['sbert_score'] >= BLUE_TO_BLUE_THRESHOLD):
                    edge = (relation['word1'], relation['word2'])
                    edge_list.append((edge, relation['sbert_score']))
                    G_disconnected.add_edge(*edge, weight=relation['sbert_score'])

        
        for word in disconnected_nodes:
            keyword_data = blue_dot_keywords_data[word]
            sentiment = keyword_data['sentiment_label']
            weight = keyword_data['contribution_weight']
            
            size = 15 + (weight * 25)
            
            tooltip = f"키워드: {word}\n감성: {sentiment}\n기여도: {weight:.3f}"
            if 'reason' in keyword_data:
                tooltip += f"\n근거: {keyword_data['reason']}"
            
            net.add_node(
                n_id=word,
                label=word,
                title=tooltip,
                color="#95a5a6",
                shape="dot",
                value=size,
                font={'size': 16, 'face': 'Pretendard', 'color': '#333333', 'bold': True},
                borderWidth=2,
                borderColor="#ffffff",
                shadow={'enabled': True, 'color': 'rgba(0,0,0,0.15)', 'size': 3}
            )

        
        if edge_list:
            for edge, weight in edge_list:
                net.add_edge(
                    edge[0], edge[1],
                    width=weight * 4,
                    color="#bdc3c7",
                    dashes=True,
                    title=f"키워드 간 연관도: {weight:.3f}"
                )
        else:
            if len(disconnected_nodes) == 1:
                pass
            elif len(disconnected_nodes) == 2:
                net.add_edge(
                    disconnected_nodes[0], disconnected_nodes[1],
                    width=1,
                    color="#bdc3c7",
                    dashes=True,
                    title="연관 관계 없음"
                )

        
        output_path = os.path.join(VISUAL_OUTPUT_DIR, f"{base_name}_disconnected_group.html")
        net.save_graph(output_path)
        self._inject_custom_scripts(output_path, f"단절된 키워드 그룹: {base_name}")
        print(f"  ✅ 저장: {output_path}")

    def _get_edge_color(self, weight):
        if weight >= 0.7:
            return "#e74c3c"
        elif weight >= 0.5:
            return "#e67e22"
        elif weight >= 0.3:
            return "#f39c12"
        else:
            return "#95a5a6"

    def _get_edge_width(self, weight):
        if weight >= 0.7:
            return 8
        elif weight >= 0.5:
            return 6
        elif weight >= 0.3:
            return 4
        else:
            return 2

    def _inject_custom_scripts(self, output_file, title):
        with open(output_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        title_div = f"<div style='position: fixed; top: 20px; left: 50%; transform: translateX(-50%); background-color: rgba(255, 255, 255, 0.95); padding: 15px 30px; border-radius: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); z-index: 999; font-family: Pretendard, sans-serif; font-size: 18px; font-weight: 600; color: #333;'>{title}</div>"
        
        html_content = html_content.replace('</head>', f'{CUSTOM_CSS}</head>')
        html_content = html_content.replace('</body>', f'{title_div}{CUSTOM_JS}</body>')

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)


class NetworkVisualizerPNG:
    def __init__(self):
        print("✅ 네트워크 시각화 준비 완료 (PNG - 레거시)")
        self.html_visualizer = NetworkVisualizerHTML()

    def visualize_single_file(self, filename: str):
        self.html_visualizer.visualize_single_file(filename)


def main():
    os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)
    
    try:
        visualizer = NetworkVisualizerHTML()
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    print("\n" + "="*60)
    print("🎯 네트워크 맵 시각화 (HTML)")
    print("="*60)
    
    choice = input("\n선택 (1: 단일 파일, 2: 전체 파일, 3: 종료): ")
    
    if choice == '1':
        filename = input("파일명 입력 (예: EB_002_finalRelation.json): ")
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

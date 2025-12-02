import json
import os
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import webbrowser
import graph_renderer 


TARGET_FILE = "output/llama3/EB_001_llama_analysis.json"
OUTPUT_HTML = "EB_001_graph_white_static.html"
MODEL_NAME = "jhgan/ko-sbert-multitask"

print(f"SBERT 모델 로딩 중({MODEL_NAME})")
embedder = SentenceTransformer(MODEL_NAME)

if not os.path.exists(TARGET_FILE):
    print(f"파일을 찾을 수 없습니다: {TARGET_FILE}")
    exit()

with open(TARGET_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

if isinstance(raw_data, dict) and "analysis_result" in raw_data:
    data_nodes = raw_data["analysis_result"]
elif isinstance(raw_data, list):
    data_nodes = raw_data
else:
    data_nodes = []

if not data_nodes:
    print("데이터가 없습니다.")
    exit()


print("데이터 분석 및 노드 배치 계산 중...")

emotions = [item.get("emotion", "무감정") for item in data_nodes]
embeddings = embedder.encode(emotions, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings, embeddings)

nx_graph = nx.Graph()
THRESHOLD = 0.60


for i in range(len(data_nodes)):
    nx_graph.add_node(i)


for i in range(len(data_nodes)):
    for j in range(i + 1, len(data_nodes)):
        score = float(cosine_scores[i][j])
        if score >= THRESHOLD:
            nx_graph.add_edge(i, j, weight=score)


print("그래프 렌더링 중")

graph_renderer.render_graph(nx_graph, data_nodes, OUTPUT_HTML)

print(f"그래프 생성 완료: {os.path.abspath(OUTPUT_HTML)}")
webbrowser.open(os.path.abspath(OUTPUT_HTML))
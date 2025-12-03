import json
import os
from pyvis.network import Network
from collections import Counter



PALETTE = ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF", "#A0C4FF", "#BDB2FF"]


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

    .legend-bar {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 15px 35px;
        border-radius: 50px;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
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


def render_graph(nx_graph, data_nodes, output_file):
    import networkx as nx 

    
    net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.set_options(json.dumps(PHYSICS_OPTIONS))

    
    components = list(nx.connected_components(nx_graph))
    legend_html = ""

    for group_idx, component in enumerate(components):
        comp_list = list(component)
        
        
        comp_emotions = [data_nodes[idx]["emotion"] for idx in comp_list]
        most_common_emotion = Counter(comp_emotions).most_common(1)[0][0]
        group_color = PALETTE[group_idx % len(PALETTE)]
        
        
        legend_html += f"<span style='margin-right: 20px; display: inline-flex; align-items: center;'><span style='color:{group_color}; font-size:24px; margin-right:5px;'>●</span> <span style='color:#333; font-weight:600;'>{most_common_emotion} 그룹</span></span>"

        
        label_id = f"group_{group_idx}"
        net.add_node(
            n_id=label_id,
            label=f"<{most_common_emotion}>", 
            shape="box",
            color={"background": "#ffffff", "border": group_color},
            font={'size': 30, 'face': 'Pretendard', 'color': group_color, 'bold': True},
            borderWidth=3,
            margin=10,
            shadow={'enabled': True, 'color': 'rgba(0,0,0,0.1)', 'size': 10, 'x': 5, 'y': 5}
        )

        
        for node_idx in comp_list:
            item = data_nodes[node_idx]
            target = item.get("target", "?")
            emotion = item.get("emotion", "")
            intensity = float(item.get("intensity", 0.5))
            summary = item.get("summary", "")

            clean_tooltip = f"Target: {target}\n감정: {emotion}\n강도: {intensity}"
            if summary:
                clean_tooltip += f"\n\n요약: {summary}"

            size = 15 + (intensity * 20) 

            net.add_node(
                n_id=node_idx,
                label=target,
                title=clean_tooltip,
                color=group_color,
                shape="dot",
                value=size,
                font={
                    'size': 18, 
                    'face': 'Pretendard',
                    'color': '#333333', 
                    'bold': True,
                    'strokeWidth': 0
                },
                borderWidth=0,
                shadow={'enabled': True, 'color': 'rgba(0,0,0,0.2)', 'size': 5} 
            )
            
            net.add_edge(label_id, node_idx, color='rgba(0,0,0,0)', width=0)

        
        for u in comp_list:
            for v in comp_list:
                if nx_graph.has_edge(u, v):
                    weight = nx_graph[u][v]['weight']
                    net.add_edge(u, v, width=weight*3, color=group_color, alpha=0.5)

    
    net.save_graph(output_file)
    _inject_custom_scripts(output_file, legend_html)

def _inject_custom_scripts(output_file, legend_html):
    with open(output_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    legend_div = f"<div class='legend-bar'>{legend_html}</div>"
    
    html_content = html_content.replace('</head>', f'{CUSTOM_CSS}</head>')
    html_content = html_content.replace('</body>', f'{legend_div}{CUSTOM_JS}</body>')

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
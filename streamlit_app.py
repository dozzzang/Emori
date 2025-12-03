import streamlit as st
from pathlib import Path
import json
import os

st.set_page_config(
    page_title="Emori",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .participant-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .participant-header h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .stExpander {
        background: white;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .stExpander:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    .stExpander > div {
        padding: 0 !important;
    }
    
    .stExpander > div > div {
        padding: 1rem 1.5rem !important;
    }
    
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    .stMarkdown {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    .stMarkdown p {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stImage"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    [data-testid="stImage"] > div {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    [data-testid="stImage"] img {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    div[data-testid="column"] > div {
        padding-top: 0 !important;
    }
    
    div[data-testid="column"] .stMarkdown:first-child {
        margin-top: 0 !important;
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #495057;
        margin-top: 0 !important;
        margin-bottom: 0.75rem !important;
        padding: 0 !important;
    }
    
    .text-content-box {
        font-size: 1.15rem;
        line-height: 1.9;
        color: #212529;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
    }
    
    .markdown-content-box {
        font-size: 1.2rem;
        line-height: 2;
        color: #212529;
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
    }
    
    .markdown-content-box h1 {
        font-size: 2rem;
        color: #667eea;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .markdown-content-box h2 {
        font-size: 1.6rem;
        color: #764ba2;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .markdown-content-box h3 {
        font-size: 1.3rem;
        color: #495057;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .markdown-content-box p {
        margin-bottom: 1.25rem;
        font-size: 1.15rem;
        line-height: 1.9;
    }
    
    .markdown-content-box ul, .markdown-content-box ol {
        margin-bottom: 1.25rem;
        padding-left: 2rem;
    }
    
    .markdown-content-box li {
        margin-bottom: 0.5rem;
        font-size: 1.15rem;
        line-height: 1.8;
    }
    
    .image-container {
        background: white;
        padding: 0;
        border-radius: 12px;
        margin: 0;
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.01);
    }
    
    .stImage {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        font-size: 1.1rem;
    }
    
    .stSidebar .stTextInput label {
        font-size: 1.3rem;
        font-weight: 600;
        color: #495057;
    }
    
    .stSidebar button {
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stSidebar button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .sidebar-section {
        margin-top: 2rem;
    }
    
    .sidebar-section h3 {
        font-size: 1.4rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 1rem;
    }
    
    .sidebar-step-list {
        list-style: none;
        padding: 0;
    }
    
    .sidebar-step-list li {
        padding: 0.75rem 0;
        font-size: 1.1rem;
        color: #6c757d;
        border-bottom: 1px solid #e9ecef;
        transition: all 0.2s ease;
    }
    
    .sidebar-step-list li:hover {
        color: #667eea;
        padding-left: 0.5rem;
    }
    
    .stExpander {
        background: white;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .stExpander:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    .stExpander label {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .stExpander > div {
        padding: 0 !important;
        animation: fadeIn 0.3s ease-out;
    }
    
    .stExpander > div > div {
        padding: 1rem 1.5rem !important;
    }
    
    .stExpander > div > div > div {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    .stMarkdown {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    .stMarkdown p {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stImage"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    [data-testid="stImage"] > div {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    [data-testid="stImage"] img {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    div[data-testid="column"] > div {
        padding-top: 0 !important;
    }
    
    div[data-testid="column"] .stMarkdown:first-child {
        margin-top: 0 !important;
    }
    
    div[data-testid="column"] [data-testid="stImage"]:first-child {
        margin-top: 0 !important;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #856404;
        font-size: 1rem;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stImage {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Emori 종합 심리 분석 대시보드</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">종합 심리 분석 및 시각화 리포트</div>', unsafe_allow_html=True)

def extract_participant_name(participant_id: str) -> str:
    """
    participant_id에서 실제 참가자 이름 추출
    예: "participant_김도단" -> "김도단", "EB_002" -> "EB_002"
    """
    if participant_id.startswith("participant_"):
        return participant_id.replace("participant_", "")
    return participant_id

def get_participant_name_from_eeg_data(participant_id: str) -> str:
    """
    EEG 데이터에서 참가자 이름 추출
    """
    try:
        eeg_json_path = Path("output/Emotion_EEG/Report_Json_Data/Report_Data.json")
        if eeg_json_path.exists():
            with open(eeg_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data:
                for key in data.keys():
                    if participant_id in key:
                        return extract_participant_name(key)
                participant_key = next(iter(data.keys()))
                return extract_participant_name(participant_key)
    except Exception:
        pass
    return None

def get_participant_id_from_name(participant_name: str) -> str:
    """
    참가자 이름으로부터 실제 participant_id (EB_002 등) 추출
    """
    import re
    
    if participant_name.startswith("EB_"):
        return participant_name
    
    try:
        output_dir = Path("output/emotionRelation/visualization")
        if output_dir.exists():
            html_files = list(output_dir.glob("*_connected_group.html"))
            if html_files:
                for html_file in html_files:
                    match = re.search(r'(EB_\d+)_connected_group\.html', html_file.name)
                    if match:
                        found_id = match.group(1)
                        found_name = get_participant_name_from_eeg_data(found_id)
                        if found_name and (found_name == participant_name or participant_name in found_name):
                            return found_id
                if html_files:
                    match = re.search(r'(EB_\d+)_connected_group\.html', html_files[0].name)
                    if match:
                        return match.group(1)
    except Exception:
        pass
    
    try:
        eeg_json_path = Path("output/Emotion_EEG/Report_Json_Data/Report_Data.json")
        if eeg_json_path.exists():
            with open(eeg_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data:
                for key in data.keys():
                    extracted_name = extract_participant_name(key)
                    if participant_name == extracted_name or participant_name in key:
                        eb_match = re.search(r'EB_\d+', key)
                        if eb_match:
                            return eb_match.group(0)
                participant_key = next(iter(data.keys()))
                eb_match = re.search(r'EB_\d+', participant_key)
                if eb_match:
                    return eb_match.group(0)
    except Exception:
        pass
    
    return "EB_002"

def load_participant_results(participant_id: str):
    """
    참가자 ID에 해당하는 모든 결과 파일 경로를 반환
    """
    results = {
        "participant_id": participant_id,
        "step1": {
            "connected_group": f"output/emotionRelation/visualization/{participant_id}_connected_group.html",
            "disconnected_group": f"output/emotionRelation/visualization/{participant_id}_disconnected_group.html",
            "final_relation": f"output/emotionRelation/finalRelation/{participant_id}_finalRelation.json"
        },
        "step2": {
            "eeg_table": None
        },
        "step3": {
            "radar_chart": "output/Emotion_EEG/Chart_Result/radar_chart.png",
            "bar_chart": "output/Emotion_EEG/Chart_Result/bar_chart.png"
        },
        "step4": {
            "generated_report": "output/Emotion_EEG/Llama3_Result/Generated_Report.txt"
        },
        "step5": {
            "wordcloud": f"output/vr_interview/visualization/{participant_id}/{participant_id}_keyword_wordcloud.png",
            "barchart": f"output/vr_interview/visualization/{participant_id}/{participant_id}_contribution_barchart.html",
            "piechart": f"output/vr_interview/visualization/{participant_id}/{participant_id}_sentiment_piechart.png",
            "summary": f"output/vr_interview/visualization/{participant_id}/{participant_id}_summary.html"
        },
        "step6": {
            "network_graph": f"{participant_id}_graph_white_static.html"
        },
        "step7": {
            "discrepancy": f"output/report_images/{participant_id}_discrepancy.png"
        },
        "step8": {
            "final_report": f"output/final_reports/{participant_id}_final_report.md"
        }
    }
    
    eeg_table_pattern = Path("output/Emotion_EEG/EEG_Tables")
    if eeg_table_pattern.exists():
        table_files = list(eeg_table_pattern.glob("*_EEG_Table.png"))
        if table_files:
            results["step2"]["eeg_table"] = str(table_files[0])
    
    return results

def display_image(path: str, caption: str = None):
    """이미지 파일을 표시"""
    if path and Path(path).exists():
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.markdown(f'<div class="warning-box">⚠️ 파일을 찾을 수 없습니다: {path}</div>', unsafe_allow_html=True)

def display_text_file(path: str):
    """텍스트 파일 내용을 표시"""
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        content_escaped = content.replace('\n', '<br>').replace('"', '&quot;').replace("'", "&#39;")
        st.markdown(f'<div class="text-content-box">{content_escaped}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warning-box">⚠️ 파일을 찾을 수 없습니다: {path}</div>', unsafe_allow_html=True)

def display_markdown_file(path: str):
    """마크다운 파일을 표시"""
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        st.markdown(f'<div class="markdown-content-box">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warning-box">⚠️ 파일을 찾을 수 없습니다: {path}</div>', unsafe_allow_html=True)

def display_html_file(path: str):
    """HTML 파일을 표시"""
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.markdown(f'<div class="warning-box">⚠️ 파일을 찾을 수 없습니다: {path}</div>', unsafe_allow_html=True)

def main():
    st.sidebar.markdown("## 설정")
    
    default_participant_id = "EB_002"
    default_participant_name = get_participant_name_from_eeg_data(default_participant_id)
    if not default_participant_name:
        default_participant_name = extract_participant_name(default_participant_id)
    if not default_participant_name or default_participant_name == default_participant_id:
        default_participant_name = "김도단"
    
    participant_name_input = st.sidebar.text_input("참가자 이름", value=default_participant_name)
    
    if st.sidebar.button("결과 불러오기", type="primary", use_container_width=True):
        st.rerun()
    
    participant_id = get_participant_id_from_name(participant_name_input)
    
    if not participant_id or participant_id == participant_name_input or not participant_id.startswith("EB_"):
        participant_id = default_participant_id
    
    results = load_participant_results(participant_id)
    
    participant_name = get_participant_name_from_eeg_data(participant_id)
    if not participant_name:
        participant_name = extract_participant_name(participant_id)
    if not participant_name or participant_name == participant_id:
        participant_name = participant_name_input if participant_name_input else participant_id
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section"><h3>분석 항목</h3></div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <ul class="sidebar-step-list">
        <li><strong>1.</strong> VR 감정 정보 기반 자기 인식 및 해석 그래프</li>
        <li><strong>2.</strong> 뇌파 데이터를 이용한 6가지 지표 시각화</li>
        <li><strong>3.</strong> 뇌파 데이터를 활용한 차트 출력</li>
        <li><strong>4.</strong> 뇌파 데이터를 바탕으로 요약 서술</li>
        <li><strong>5.</strong> 감정 빈도 분석 그래프 및 주요 표현</li>
        <li><strong>6.</strong> 학생 인터뷰 기반 감정의 원인 그래프</li>
        <li><strong>7.</strong> 뇌파 - 인터뷰 일치도 확인</li>
        <li><strong>8.</strong> 최종 요약 서술</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="participant-header"><h2>참가자: {participant_name}</h2></div>', unsafe_allow_html=True)
    
    step_names = [
        ("1. VR 감정 정보 기반 자기 인식 및 해석 그래프", "step1"),
        ("2. 뇌파 데이터를 이용한 6가지 지표 시각화", "step2"),
        ("3. 뇌파 데이터를 활용한 차트 출력", "step3"),
        ("4. 뇌파 데이터를 바탕으로 요약 서술", "step4"),
        ("5. 감정 빈도 분석 그래프 및 주요 표현", "step5"),
        ("6. 학생 인터뷰 기반 감정의 원인 그래프", "step6"),
        ("7. 뇌파 - 인터뷰 일치도 확인", "step7"),
        ("8. 최종 요약 서술", "step8")
    ]
    
    for idx, (step_title, step_key) in enumerate(step_names, 1):
        with st.expander(step_title, expanded=True):
            if step_key == "step1":
                with st.expander("VR 감정 정보 기반 자기 인식 및 해석 그래프 설명", expanded=False):
                    st.markdown("""
                    <div class="info-section">
                        <div class="info-item">
                            <div class="info-item-desc">대상 학생이 선택한 감정(중앙)과, 인터뷰 기반 추출 감정(그외)과의 연관성을 분석합니다. 인터뷰 기반 추출 감정과 대상 학생이 선택한 감정이 가까울수록 해당 감정단어가 학생이 선택한 감정에 큰 영향을 미치는 결과를 보여줍니다.</div>
                        </div>
                        <div class="info-item">
                            <div class="info-item-desc">대상 학생이 선택한 감정과 연관성이 적은 감정은 disconnected_group에 따로 나타내집니다.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<p class="section-title">연결된 핵심 표현 그룹</p>', unsafe_allow_html=True)
                    display_html_file(results[step_key]["connected_group"])
                with col2:
                    st.markdown('<p class="section-title">단절된 핵심 표현 그룹</p>', unsafe_allow_html=True)
                    display_html_file(results[step_key]["disconnected_group"])
            
            elif step_key == "step2":
                st.markdown('<p class="section-title">뇌파 수준별 색상 테이블</p>', unsafe_allow_html=True)
                
                with st.expander("뇌파 시각화 설명", expanded=False):
                    st.markdown("""
                    <div class="info-section">
                        <div class="info-item">
                            <div class="info-item-title">1. 색의 농도</div>
                            <div class="info-item-desc">색이 진할수록 해당 뇌파 반응이 '강하게' 나타난 것입니다.</div>
                        </div>
                        <div class="info-item">
                            <div class="info-item-title">2. 시간의 흐름 (행)</div>
                            <div class="info-item-desc">위(STEP2)에서 아래(STEP4)로 내려갈수록 체험이 진행되면서 어떻게 변화했는지 보여줍니다.</div>
                        </div>
                        <div class="info-item">
                            <div class="info-item-title">3. 지표의 구분 (열)</div>
                            <div class="info-item-desc">각 색깔은 서로 다른 뇌파를 나타냅니다.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                display_image(results[step_key]["eeg_table"])
            
            elif step_key == "step3":
                with st.expander("Radar Chart 설명", expanded=False):
                    st.markdown("""
                    <div class="info-section">
                        <div class="info-item">
                            <div class="info-item-title">인지 부하</div>
                            <div class="info-item-desc">뇌가 스트레스를 받고 긴장하여 정신적으로 얼마나 힘이 들고 지쳐 있는지를 나타내는 '두뇌 피로도' 입니다.</div>
                        </div>
                        <div class="info-item">
                            <div class="info-item-title">정서적 긍정성</div>
                            <div class="info-item-desc">부정적인 스트레스를 걷어내고, 흥미와 즐거운 자극을 통해 얼마나 '기분 좋은 상태'인지를 보여줍니다.</div>
                        </div>
                        <div class="info-item">
                            <div class="info-item-title">주도적 집중</div>
                            <div class="info-item-desc">단순히 쳐다보는 것을 넘어, 얼마나 '적극적이고 능동적으로' 집중하고 있는지를 의미합니다.</div>
                        </div>
                        <div class="info-item">
                            <div class="info-item-title">이완-활력 균형</div>
                            <div class="info-item-desc">너무 축 처지지도, 너무 들뜨지도 않고 마음이 얼마나 '안정적이고 조화로운 에너지 상태'를 유지하는지 보여줍니다.</div>
                        </div>
                        <div class="info-item">
                            <div class="info-item-title">종합 몰입도</div>
                            <div class="info-item-desc">흥미와 집중, 참여 의지를 모두 종합하여 현재 활동에 얼마나 '깊이 빠져들어 있는지'를 나타내는 최종 점수입니다.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<p class="section-title">방사형 차트</p>', unsafe_allow_html=True)
                    display_image(results[step_key]["radar_chart"])
                with col2:
                    st.markdown('<p class="section-title">막대 그래프</p>', unsafe_allow_html=True)
                    display_image(results[step_key]["bar_chart"])
            
            elif step_key == "step4":
                st.markdown('<p class="section-title">뇌파 기반 요약 보고서</p>', unsafe_allow_html=True)
                display_text_file(results[step_key]["generated_report"])
            
            elif step_key == "step5":
                with st.expander("감정 빈도 분석 그래프 및 주요 표현 설명", expanded=False):
                    st.markdown("""
                    <div class="info-section">
                        <div class="info-item">
                            <div class="info-item-title">중요도 차트</div>
                            <div class="info-item-desc">인터뷰 내용을 기반으로한 대상 학생의 큰 분류(긍정/중립/부정)의 감정을 분석합니다. 해당 막대그래프는 결과로 나오는 최종 큰 분류(긍정/중립/부정)의 감정에 어떤 단어가 크게 기여했는지 결과를 보여주며, 결과로 나오지 않은 다른 큰 분류(긍정/중립/부정)의 감정에 기여한 단어 또한 보여줍니다.</div>
                        </div>
                        <div class="info-item">
                            <div class="info-item-title">워드클라우드</div>
                            <div class="info-item-desc">중요도차트를 기반으로한 기여 감정들을 한눈에 보기 쉽게 워드크라우드 형태로 표현합니다. 색상에 따른 다양한 감정과 단어 크기에 따라 현재 감정에 기여하는 정도가 큽니다.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<p class="section-title">주요 표현 모음</p>', unsafe_allow_html=True)
                    display_image(results[step_key]["wordcloud"])
                with col2:
                    st.markdown('<p class="section-title">중요도 분석 차트</p>', unsafe_allow_html=True)
                    display_html_file(results[step_key]["barchart"])
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown('<p class="section-title">감정 분포 그래프</p>', unsafe_allow_html=True)
                    display_image(results[step_key]["piechart"])
                with col4:
                    st.markdown('<p class="section-title">인터뷰 요약</p>', unsafe_allow_html=True)
                    display_html_file(results[step_key]["summary"])
            
            elif step_key == "step6":
                with st.expander("학생 인터뷰 기반 감정의 원인 그래프 설명", expanded=False):
                    st.markdown("""
                    <div class="info-section">
                        <div class="info-item">
                            <div class="info-item-desc">실제 학생 인터뷰를 토대로 모든 문맥을 파악하여 학생이 실제로 어떤 정서를 느끼고 있는지 비슷한 감정 그룹을 묶어 감정의 정도와 원인을 한 눈에 보도록 나타내는 그래프를 보여줍니다.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<p class="section-title">학생 인터뷰 기반 감정의 원인 그래프</p>', unsafe_allow_html=True)
                display_html_file(results[step_key]["network_graph"])
            
            elif step_key == "step7":
                st.markdown('<p class="section-title">뇌파-인터뷰 일치도 확인</p>', unsafe_allow_html=True)
                display_image(results[step_key]["discrepancy"])
            
            elif step_key == "step8":
                st.markdown('<p class="section-title">최종 종합 보고서</p>', unsafe_allow_html=True)
                display_markdown_file(results[step_key]["final_report"])

if __name__ == "__main__":
    main()

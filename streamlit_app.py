import streamlit as st

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
    
    .welcome-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 3rem;
        border-radius: 12px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Emori</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">청소년 정서 종합 분석 시스템</div>', unsafe_allow_html=True)

st.markdown("""
<div class="welcome-box">
    <h2>환영합니다!</h2>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        Emori는 뇌파 데이터와 인터뷰 내용을 연관 분석하여<br>
        청소년의 정서를 종합적으로 파악하는 시스템입니다.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>파일 업로드 및 분석</h3>
        <p>뇌파 데이터와 인터뷰 파일을 업로드하여 새로운 분석을 시작하세요.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("파일 업로드 페이지로 이동", use_container_width=True, type="primary"):
        st.switch_page("pages/1_파일_업로드_및_분석.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>분석 결과 대시보드</h3>
        <p>이미 완료된 분석 결과를 확인하고 시각화 자료를 탐색하세요.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("대시보드로 이동", use_container_width=True, type="secondary"):
        st.switch_page("pages/0_대시보드.py")

st.markdown("---")

st.markdown("""
### 주요 기능

1. **VR 감정 정보 기반 자기 인식 및 해석 그래프**
   - 뇌파 메인 감정과 인터뷰 감정 간 연관성 분석

2. **뇌파 데이터 시각화**
   - 6가지 뇌파 지표 색상 테이블 및 Radar Chart

3. **뇌파 데이터 요약**
   - LLaMA3 모델 기반 자동 요약 보고서 생성

4. **감정 빈도 분석**
   - 인터뷰 기반 감정 키워드 중요도 분석 및 워드클라우드

5. **감정 원인 그래프**
   - 토픽 네트워크맵을 통한 감정 간 연관성 분석

6. **뇌파-인터뷰 일치도 확인**
   - 두 데이터 간 일치도 정량적 분석

7. **최종 종합 보고서**
   - 모든 분석 결과를 통합한 최종 보고서 자동 생성
""")

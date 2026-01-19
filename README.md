## 소개

**뇌파(EEG) 데이터**와 **VR 인터뷰 텍스트**를 함께 분석하여  
정서 상태를 정량적으로 파악하고, **시각화·요약·연관성 분석**을 한 번에 제공하는 웹 대시보드

## 배경 및 목표

- 초·중·고 청소년 정서 파악을 위한 뇌파 측정 및 VR 기반 감정 데이터 수집 시스템 보유
- 뇌파 데이터와 상담 텍스트 분석 시 상담사의 **주관적 해석 의존**이라는 한계 존재
- 이를 해소하기 위해 **정서·뇌파 데이터 기반 유의미 연관분석 알고리즘**과  
  **자동 요약·시각화 대시보드**를 개발하여 객관적이고 재현 가능한 분석을 지원

## 사용 방법

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

- 기본 실행 엔트리포인트 : `streamlit_app.py`
- 권장 환경 : Python 3.10+

## 주요 기능 및 대응 파일

- **분석·연관성 시각화**
  - **VR 감정 기반 자기 인식/해석 그래프**
    - 파일: `src/emotion-interview_relation/step3_relation_analyzer.py`, `src/emotion-interview_relation/step4_visualizer.py`, `src/graph_renderer.py`
  - **뇌파 메인 감정 × 인터뷰 감정 네트워크 그래프**
    - 파일: `src/emotion-interview_relation/step3_relation_analyzer.py`, `src/emotion-interview_relation/step4_visualizer.py`
  - **뇌파-인터뷰 정서 일치도 분석 및 시각화**
    - 파일: `src/Emotion_EEG/Run/run_pipeline.py`, `src/graph_renderer.py`

- **뇌파 지표 시각화**
  - **6가지 뇌파 지표 색상 테이블**  
    (인지 부하, 정서적 긍정성, 주도적 집중, 이완-활력 균형, 종합 몰입도 등)
    - 파일: `src/Emotion_EEG/EEG_Color/EEG_Table_Visualizer.py`
  - **Radar Chart & 막대 그래프 기반 종합 분석**
    - 파일: `src/Emotion_EEG/Rader_Chart/RaderChart.py`, `src/make_graph.py`

- **텍스트·감정 분석**
  - **감정 빈도 분석, 키워드 중요도, 워드클라우드, 감정 분포 파이 차트**
    - 파일: `src/emotion-interview_relation/step2_keyword_extractor.py`, `src/emotion-interview_relation/step4_visualizer.py`
  - **학생 인터뷰 기반 감정 원인 네트워크(토픽 네트워크맵)**
    - 파일: `src/emotion-interview_relation/step3_relation_analyzer.py`, `src/emotion-interview_relation/step4_visualizer.py`

- **요약·보고서**
  - **LLaMA3 파인튜닝 기반 뇌파 데이터 서술형 요약**
    - 파일: `src/Emotion_EEG/DescriptiveSummary_Llama3/Llama3Main.py`, `src/Emotion_EEG/DescriptiveSummary_Llama3/Llama3.py`, `src/FinalReportGenerator.py`
  - **모든 결과를 통합한 최종 종합 보고서 자동 생성**
    - 파일: `src/FinalReportGenerator.py`

## 기대효과

- 뇌파와 인터뷰 내용의 정량적 연관성을 시각화해 상담사의 객관적 분석 지원
- 인터뷰 자동 분석으로 학생의 주된 정서를 문맥 기반으로 파악
- 모든 분석 결과를 통합 대시보드로 제공해 효율적 아카이빙 및 분석 가능

## 실적 산출

- 2025 추계종합학술대회 논문 제출 및 발표: “정서, 뇌파 데이터와 상담 텍스트의 다중모달 연관분석 기반 심리상태 진단시스템” 동상 수상
- 모델 성능 확보: 서술형 요약 파인튜닝 7 Epoch 학습 결과 손실 2.91 → 0.35 수렴, 토큰 예측 정확도 약 48% → 91% 달성

## 기술 스택

- 프레임워크/라이브러리: Python 3.10+, Streamlit, PyTorch (Transformers, SentenceTransformer), NumPy, Pandas, scikit-learn
- 모델/아키텍처: LLaMA3 (Meta-Llama-3.1-8B-Instruct, Groq API 추론), SentenceTransformer(jhgan/ko-sbert-multitask, jhgan/ko-sroberta-multitask)
- NLP: MeCab 한국어 형태소 분석

## 주요 저장소 구조

- `streamlit_app.py`: 대시보드 진입점
- `pages/`: 스트림릿 멀티페이지 구성
- `src/Emotion_EEG/`: 뇌파 데이터 전처리·시각화·요약 모듈
- `src/emotion-interview_relation/`: 인터뷰 텍스트 감정 추출·연관성·시각화 파이프라인
- `output/Emotion_EEG/`: 학습/추론 결과 및 보고서 샘플
- `data/`: 원천 데이터 및 레이블 예시

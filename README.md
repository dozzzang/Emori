# 과정 진행 절차
1단계 : HWP → TXT 변환 ✅ <br>
2단계 : 형태소 분석 (명사/동사/형용사 추출) ✅ <br>
3단계 : 감정 분석 (BERT 모델) ✅ <br>
4단계 : 키워드 추출 (TF-IDF, TextRank) <br>
5단계 : 감정 단어 우선 정렬 <br>
6단계 : 시각화 (워드클라우드, 네트워크) <br>
7단계 : 패턴 분석 및 인사이트 도출 <br>


# 1단계 : HWP → TXT 변환 자동화 ✅ (완료) <br>
작업 내용: <br>
* olefile 라이브러리를 사용한 HWP 5.0 파일 파싱 <br>
* 3가지 텍스트 추출 방법 구현: <br>
    1. HWP 레코드 구조 파싱 (Tag ID 67) <br>
    2. UTF-16LE 디코딩 <br>
    3. 바이트 패턴 검색 (한글 유니코드 범위) <br>
* zlib 압축 해제 지원 <br>
* 변환 품질 평가 (한글 문자 개수 기반) <br>

완성된 파일: <br>
* step1_hwp_to_txt_olefile.py - 메인 변환 프로그램 <br>
* manual_txt_organizer.py - 수동 변환 지원 도구 <br>
GitHub 상태: <br>

* 브랜치: feature/step1-hwp-converter <br>
환경 설정: <br>
* 가상환경(venv) 생성 완료 <br>

* 필요 라이브러리: olefile 설치 완료 <br>


# 2단계 : 형태소 분석 (Mecab) ✅ (완료) <br>
작업 내용: <br>
* Mecab 형태소 분석기 적용 <br>
* Apple silicon / Intel Mac 자동 경로 감지 기능 추가 <br>
* 명사 (NNG, NNP), 동사 (VV), 형용사(VA) 추출 <br>
* 단어 빈도 분석 기능 구현 <br>
* 불용어 제거 기능 추가 <br>
* 분석 결과를 JSON 형식으로 저장 (output/morpheme 폴더 구조 추가) <br>

완성된 파일 : <br>
* step2_mecab_analyzer.py : 형태소 분석 메인 프로그램 <br>

GitHub 상태 : <br>
* 브랜치: feature/step2-morpheme-analysis <br>

환경설정 : <br>
* python-mecab-ko 설치 (Mac 환경 자동 경로 감지 지원) <br>
* 필요 라이브러리 : json, collections, re <br>

# 3단계 : 감정 분석 (사전 + BERT) ✅ (완료) <br>
목적 :
* 인터뷰 텍스트 감정을 자동으로 분석하여 긍정 / 부정 / 중립 분류
* 두가지 방식으로 감정을 분석하며, 정확성 및 속도 확보
    1. KNU 감정사전 기반 분석 (규칙기반 / 빠름)
    2. BERT 딥러닝 기반 분석 (문맥 이해 / 정확)

    방법 1 : KNU 감정사전 기반 분석
    * 14000+ 감정 단어 사전 불러옴
        * "좋다" -> +1.0
        * "행복하다" -> +0.8
        * "불안하다" -> -0.7
    * 텍스트 단어 매칭 및 점수 계산
        * "상담이 좋았고 편안했어요" -> "좋다"(+1,0) + "편안하다"(+0.8) -> 평균점수 = (1.0 + 0.8) / 2 = 0.9 -> 긍정
    
    방법 2 : BERT 딥러닝 모델 기반 분석 (핵심 AI)
    BERT란?
    * google 개발 Transformer 기반 자연어 이해 모델
    * 문장을 양방향으로 분석하여 의미와 감정까지 파악
    * 한국어 문법·문맥 학습 (사전 학습 데이터: 위키백과, 뉴스, 커뮤니티)
    * 약 1억 1천만 개 파라미터 (110M)

    BERT 내부 처리 단계
    1. 토큰화 (Tokenization)
        * 문장을 서브워드 단위로 분해하여 숫자 ID로 변환
            * “상담사님과 이야기하니 기분이 좋아졌어요” -> [CLS] 상담 ##사 ##님 ##과 이야기 ##하 ##니 기분 ##이 좋 ##아 ##졌 ##어요 [SEP]
        * 이유
            * 미등록 단어 처리 가능
            * "좋아졌어요” → “좋” + “아” + “졌” + “어요” 로 의미 세분화
    
    2. BERT 인코딩 (Transformer Layers)
        * Embedding Layer: 각 토큰을 768차원 벡터로 변환
        * 12개 Transformer Layer: Self-Attention을 통해 문맥적 관계 파악
            * Layer 1–4: 단어 의미
            * Layer 5–8: 구문 관계
            * Layer 9–12: 문맥/감정 이해
        * 출력: [CLS] 벡터 (768차원) → 문장 전체 의미 표현

    3. Self-Attention 메커니즘 (핵심)
        * ex : “VR 체험 후 불안감이 많이 줄어들었습니다”
            * 단어 | attention점수 | 의미
            * 불안감 | 1.0 | 자기자신
            * 줄어들 | 0.8 | 핵심 관계
            * 많이 | 0.3 | 강조
            -> “불안감”과 “줄어들었다”의 관계를 파악 -> 긍정 감정
    
    4. 감정 분류 (Classification Layer)
        * [CLS] 벡터 (768차원)와 가중치 내적 연산
        * Softmax를 통해 긍정 / 부정 / 중립 확률 계산
        * ex : 긍정 확률 = 0.82 (82%) → 감정: 긍정

    BERT 학습 과정
    1. 사전 학습 (Pre-training)
        * 한국어 위키백과, 뉴스, 커뮤니티 수억 문장 학습
        * Masked Language Model (MLM) 방식
            * ex : “나는 [MASK]를 먹었다” → [MASK] = “밥” 예측 학습
            * 결과 : 문법, 어휘, 문맥 이해
    
    2. 파인튜닝 (Fine-tuning)
        * 감정 레이블 데이터로 재학습
            * "이 영화 정말 좋아요" → 긍정
            * "완전 짜증나네요" → 부정  
            * "그냥 그랬어요" → 중립
        * BERT 출력 + Dense Layer로 감정 분류 최적화

    사용 모델
    * 모델명 : matthewburke/korean_sentiment
    * 기반 모델 : KcELECTRA
    * 구조 :
        * Layers: 12
        * Hidden size: 768
        * Attention heads: 12
        * 파라미터 수: 110M
        * 모델 크기: 498MB

    기술 스택
        from transformers import pipeline
        import torch

        analyzer = pipeline(
            task="sentiment-analysis",
            model="matthewburke/korean_sentiment",
            tokenizer="matthewburke/korean_sentiment"
        )

        result = analyzer("상담이 좋았어요") # {'label': 'POSITIVE', 'score': 0.876}
        

    작업 내용 요약
    * 사전 기반 분석 : 14,000+ 감정 단어 자동 매칭, 평균 점수 계산
    * BERT 모델 분석 : 한국어 사전학습 모델 적용, 문맥 기반 감정 예측
    * 통합 기능 : 두 결과 비교 / 감정 분포 통계 / JSON 저장

    실행 방법
        # 필수 라이브러리 설치
        pip install transformers torch

        # 실행
        python step3_sentiment_analysis_complete.py

        # 1. 분석 방법 선택 (1: 사전, 2: 사전+BERT)
        선택: 2

        # 2. 분석 모드 선택 (1: 단일, 2: 전체 파일)
        선택: 2

    

# 과정 진행 절차
1단계 : HWP → TXT 변환 ✅ <br>
2단계 : 형태소 분석 (명사/동사/형용사 추출) ✅ <br>
3단계 : 감정 분석 (BERT 모델) ✅ <br>
4단계 : 키워드 추출 (TF-IDF, TextRank) ✅ <br>
5단계 : 감정 단어 우선 정렬 ✅ <br>
6단계 : 시각화 (워드클라우드, 네트워크) ✅ <br>


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

# 4단계 : 키워드 추출 (TF-IDF & TextRank) ✅ (완료) <br>
* 목적 
    * 인터뷰 텍스트에서 핵심 키워드 자동 추출
    * 3가지 알고리즘으로 다각적 분석 수행
        * 빈도 기반 (Frequency) - 가장 직관적
        * TF-IDF (Term Frequency-Inverse Document Frequency) - 문서별 중요도 고려
        * TextRank (그래프 기반) - 단어 간 관계 고려

* 기술 스택
    * 핵심 라이브러리
        * scikit-learn (sklearn)
            * TfidfVectorizer: TF-IDF 벡터화 및 가중치 계산
            * 최대 1000개 단어 추출
            * min_df=1 (최소 1개 문서), max_df=0.8 (80% 이상 문서 제외)
        * NetworkX (nx)
            * 그래프 기반 TextRank 구현
            * 노드: 단어들
            * 엣지: 동시 등장 관계 (가중치: 동시 등장 횟수)
            * PageRank 알고리즘 적용
        * NumPy (np)
            * 수치 연산 지원

* 데이터 처리
    * collections.Counter: 단어 빈도 계산
    * json: 결과 저장/로드
    * pathlib.Path: 파일 경로 처리
    * os: 폴더/파일 관리

* 알고리즘 상세 설명
    1. 빈도 기반 (Frequency)
        방식: 단어가 얼마나 많이 등장하는가?
        예시: "불안" 10회, "개선" 8회, "기분" 6회
        장점: 빠르고 직관적
        단점: 맥락을 고려하지 않음

    2. TF-IDF (Term Frequency-Inverse Document Frequency)
        개념: 특정 문서에서 특별한 단어를 찾는다
        공식: TF-IDF = (단어 빈도) × log(전체문서 수 / 단어포함문서 수)

        예시 (3개 문서):
        - 문서1: "상담", "효과적", "개선" 등
        - 문서2: "상담", "효과적", "변화" 등
        - 문서3: "상담", "효과적", "만족" 등

        "상담" = 높은 TF, 낮은 IDF → 낮은 TF-IDF (공통단어)
        "개선" = 중간 TF, 높은 IDF → 높은 TF-IDF (구분 가능)

        처리 과정:
        1. 토큰화 (각 문서의 단어들을 수치 벡터로 변환)
        2. TF 계산 (각 단어의 빈도)
        3. IDF 계산 (문서 전체에서의 희소성)
        4. TF-IDF 스코어링 (상품-합산)

    3. TextRank (그래프 기반)
        개념: 단어들 사이의 관계를 그래프로 모델링하고 중요도 계산
        알고리즘: PageRank (구글 검색의 페이지 순위 알고리즘)

        처리 단계:
        1. 그래프 생성
        - 노드(Node): 추출된 명사들
        - 엣지(Edge): 윈도우 크기 내 동시 등장
        
        2. 가중치 할당
        - 같이 자주 등장하는 단어 쌍: 가중치↑
        - 예: "VR" ↔ "체험" (자주 함께 등장)
        
        3. PageRank 계산
        - 중요한 단어: 다른 중요 단어와 연결
        - "VR"이 중요하면 "VR"과 연결된 단어도 중요성↑

        장점: 단어 간 관계와 맥락을 함께 고려

* 입력 데이터
    * 출처: output/morpheme/ 폴더의 형태소 분석 JSON 파일
    * 필수 필드: all_noun : 추출된 명사 리스트 (키워드 추출에 사용)

* 출력 데이터
    * 개별 파일 결과
        * 파일: output/keywords/{파일명}_keywords.json
            {
                "filename": "EG_001.txt",
                "total_nouns": 245,
                "frequency_keywords": [
                    ["불안", 10],
                    ["개선", 8],
                    ["기분", 6]
                ],
                "tfidf_keywords": [
                    ["효과", 0.4521],
                    ["변화", 0.3847],
                    ["만족", 0.3214]
                ],
                "textrank_keywords": [
                    ["VR", 0.0854],
                    ["체험", 0.0721],
                    ["상담", 0.0698]
                ]
            }

    * 전체 요약 결과
        * 파일: output/keywords/keywords_summary.json
            {
                "total_files": 50,
                "overall_top_keywords": [
                    ["상담", 450],
                    ["개선", 380],
                    ["만족", 320],
                    ["불안", 290]
                ]
            }

* 코드 구조
    * 주요 클래스 : keywordExtractor
        * 매서드 1 : extract_frequency_keywords()
            * 단어 빈도 기반 추출
            * 속도: 가장 빠름 (O(n))
            * 정확도: 기본 수준

        * 메서드 2: extract_tfidf_keywords()
            * TF-IDF 기반 추출
            * 최소 2개 이상 문서 필요
            * 문서별 고유한 특성 단어 발견 우수
            * 벡터 크기: 최대 1000개 단어

        * 메서드 3: extract_textrank_keywords()
            * TextRank(PageRank) 기반 추출
            * 최소 5개 이상 단어 필요
            * 윈도우 크기(기본값: 5) 내 동시 등장 관계 분석
            * 그래프 가중치로 반복 업데이트

        * 메서드 4: analyze_single_file()
            * 단일 파일 분석
            * 3가지 알고리즘 모두 적용
            * 명사만 사용 (top_n=10)

        * 매서드 5: analyze_all_files()
            * 모든 파일 일괄 분석
            * 개별 파일 결과 저장
            * TF-IDF 전체 문서 대상 계산
            * 전체 통계 요약 생성

* 실행방법
    * 필수 라이브러리 설치
        pip install scikit-learn networkx numpy
    
    * 스크립트 실행
        python step4_keyword_extraction.py


# 5단계 : 감정 단어 우선 정렬 ✅ (완료) <br>
* 목적
    * 형태소 분석된 명사에서 감정사전을 이용해 감정 단어만 추출
    * 감정 단어를 빈도수로 수치화하여 가장 많이 사용된 단어부터 정렬
    * 긍정/부정 단어로 자동 분류
    * Step 6 시각화를 위한 데이터 준비

* 기술 스택
    * 감정사전: KNU SentiWord Dict (6,461개 단어)
    * 빈도 분석: collections.Counter
    * 데이터 처리: JSON 형식 저장
    * 지원 포맷: JSON, CSV, TXT (탭/공백 구분)

* 핵심 알고리즘
    1. 감정 단어 추출
        입력: 형태소 분석된 명사 리스트
            ["상담", "불안", "개선", "만족", "효과", ...]

        처리:
        1. 각 명사를 감정사전과 매칭
        - "불안" → 발견 (감정사전에 있음)
        - "상담" → 미발견 (감정사전에 없음)

        2. 감정 단어만 필터링
        감정 단어: ["불안", "개선", "만족", "효과"]

        3. 빈도수 계산
        불안: 5회
        개선: 3회
        만족: 2회
        효과: 1회
    
    2. 빈도수 기준 정렬
        Counter를 사용한 자동 정렬:
        most_common() → [(단어, 빈도), ...]

        결과:
        [("불안", 5), ("개선", 3), ("만족", 2), ("효과", 1)]

    3. 긍정/부정 분류
        감정사전의 점수 기준:
        - 양수(+) → 긍정 단어
        - 음수(-) → 부정 단어

        예시:
        개선: +0.85 → 긍정
        불안: -0.90 → 부정

* 입력 데이터
    * 경로: output/morpheme/
    * 파일명: {파일명}_morpheme.json
    * 필수 필드: all_nouns (형태소 분석된 명사 리스트)

* 출력 데이터
    * 개별 파일 결과
        * 파일: output/emotion_ranking/{파일명}_emotion_ranking.json
            {
                "filename": "EG_001.txt",
                "total_nouns": 245,
                "emotion_words_count": 28,
                "emotion_words_frequency": 45,
                "positive_frequency": 18,
                "negative_frequency": 27,
                "emotion_words": [
                ["불안", 5],
                ["개선", 3],
                ["만족", 2],
                ["효과", 1]
                ],
                "positive_words": [
                ["개선", 3],
                ["만족", 2],
                ["효과", 1]
                ],
                "negative_words": [
                ["불안", 5],
                ["피로", 2],
                ["갈등", 1]
                ]
            }

* 실행 방법
    * 감정사전 준비
        # KNU 감정사전 자동 다운로드 (선택)
        python knu_sentiment_lexicon.py
        # → data/sentiment/SentiWord_Dict.txt에 저장됨

* 상호 작용
    😊 5단계: 감정 단어 우선 정렬

    📖 기본 감정사전 사용: data/sentiment/SentiWord_Dict.txt
    📖 감정사전 로드 중: data/sentiment/SentiWord_Dict.txt
    ✅ 6461개 단어 로드 완료

    분석 모드 선택:
    1. 단일 파일 분석
    2. 전체 파일 분석

    선택 (1-2): 2

# 6단계 : 감정 단어 시각화 ✅ (완료)

* 목적
    * Step 5에서 추출한 감정 단어 데이터를 시각화
    * 워드클라우드, 막대 그래프, 파이 차트로 감정 분석 결과 표현
    * 긍정/부정 단어 비교 분석
    * 전체 감정 경향 한눈에 파악

* 기술 스택
    * 시각화 라이브러리
        * matplotlib: 그래프, 차트
        * wordcloud: 워드클라우드
        * numpy: 수치 연산

* 생성되는 시각화
    1. 워드클라우드
        특징:
        - 단어 크기 = 빈도수
        - 색상: 빨강(부정) → 초록(긍정)
        - 고해상도 (300 dpi)

        파일명: {파일명}_wordcloud.png

    2. 막대 그래프 (모든감정단어)
        특징:
        - 상위 15개 감정 단어
        - 가로 막대 그래프
        - 각 막대에 빈도 수치 표시

        파일명: {파일명}_all_barchart.png

    3. 파이차트 (긍정/부정비율)
        특징:
        - 긍정 vs 부정 비율
        - 퍼센티지 표시
        - 범례에 개수 표시

        파일명: {파일명}_pie.png

    4. 비교차트 (긍정/부정단어)
        특징:
        - 좌측: 상위 10개 긍정 단어
        - 우측: 상위 10개 부정 단어
        - 각각의 빈도 표시

        파일명: {파일명}_comparison.png
    
* 실행 방법
    * 필수 라이브러리 설치
        pip install matplotlib wordcloud numpy
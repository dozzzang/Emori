# 과정 진행 절차
1단계 : HWP → TXT 변환 ✅ <br>
2단계 : 형태소 분석 (명사/동사/형용사 추출) <br>
3단계 : 감정 분석 (BERT 모델) <br>
4단계 : 키워드 추출 (TF-IDF, TextRank) <br>
5단계 : 감정 단어 우선 정렬 <br>
6단계 : 시각화 (워드클라우드, 네트워크) <br>
7단계 : 패턴 분석 및 인사이트 도출 <br>


# 1단계 
1단계 : HWP → TXT 변환 자동화 ✅ (완료) <br>
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
* PR 생성 대기 중 <br>
환경 설정: <br>
* 가상환경(venv) 생성 완료 <br>
* 필요 라이브러리: olefile 설치 완료 <br>

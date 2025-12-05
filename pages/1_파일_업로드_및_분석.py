import streamlit as st
from pathlib import Path
import re
import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="파일 업로드 및 분석 - Emori",
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
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #667eea;
        margin: 2rem 0;
        text-align: center;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #155724;
    }
    
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #721c24;
    }
    
    .info-box {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #0c5460;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def extract_participant_id_from_eeg(content: str) -> str:
    """뇌파 파일에서 참가자 ID 추출"""
    # NAME 필드에서 추출
    name_match = re.search(r'NAME\s*:\s*(\S+)', content)
    if name_match:
        name = name_match.group(1).strip()
        # EB_002 형식인지 확인
        eb_match = re.search(r'([A-Z]{2}_\d{3})', name)
        if eb_match:
            return eb_match.group(1)
        return name
    return None

def save_uploaded_file(uploaded_file, target_path: Path):
    """업로드된 파일을 지정된 경로에 저장"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return target_path

def run_analysis(participant_id: str):
    """main.py의 main 함수를 실행하여 분석 수행"""
    try:
        from src.main import main
        
        # 출력을 캡처하기 위한 StringIO
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            main(participant_id)
        
        output_text = output_buffer.getvalue()
        error_text = error_buffer.getvalue()
        
        return True, output_text, error_text
    except Exception as e:
        return False, str(e), None

def main():
    st.markdown('<h1 style="text-align: center; color: #667eea;">파일 업로드 및 분석</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.1rem;">뇌파 데이터와 인터뷰 파일을 업로드하여 분석을 시작하세요</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 안내 정보
    st.markdown("""
    <div class="info-box">
        <strong>업로드 가이드</strong><br>
        • 뇌파 파일: RECORD*.txt 형식의 뇌파 측정 데이터 파일<br>
        • 인터뷰 파일: .txt 형식의 학생 인터뷰 내용 파일<br>
        • 파일명은 참가자 ID(예: EB_002)를 포함해야 합니다
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 뇌파 데이터 파일")
        eeg_file = st.file_uploader(
            "뇌파 파일 선택 (RECORD*.txt)",
            type=['txt'],
            key='eeg_file',
            help="뇌파 측정 기기로부터 생성된 RECORD*.txt 파일을 업로드하세요"
        )
        
        if eeg_file:
            st.success(f"파일 선택됨: {eeg_file.name}")
            # 파일 내용 미리보기
            content = eeg_file.read().decode('utf-8')
            eeg_file.seek(0)  # 파일 포인터 리셋
            
            # 참가자 ID 추출 시도
            participant_id = extract_participant_id_from_eeg(content)
            if participant_id:
                st.info(f"추출된 참가자 ID: **{participant_id}**")
    
    with col2:
        st.markdown("### 인터뷰 데이터 파일")
        interview_file = st.file_uploader(
            "인터뷰 파일 선택 (.txt)",
            type=['txt'],
            key='interview_file',
            help="학생 인터뷰 내용이 담긴 .txt 파일을 업로드하세요"
        )
        
        if interview_file:
            st.success(f"파일 선택됨: {interview_file.name}")
            # 파일명에서 참가자 ID 추출 시도
            file_stem = Path(interview_file.name).stem
            eb_match = re.search(r'([A-Z]{2}_\d{3})', file_stem)
            if eb_match:
                extracted_id = eb_match.group(1)
                st.info(f"추출된 참가자 ID: **{extracted_id}**")
    
    st.markdown("---")
    
    # 참가자 ID 직접 입력 옵션
    st.markdown("### 참가자 ID 확인")
    manual_id = st.text_input(
        "참가자 ID를 직접 입력하거나 위에서 자동 추출된 ID를 확인하세요",
        value=participant_id if 'participant_id' in locals() and participant_id else "",
        help="예: EB_002, EG_001 등"
    )
    
    # 분석 시작 버튼
    st.markdown("---")
    
    if st.button("분석 시작", type="primary", use_container_width=True):
        # 유효성 검사
        if not eeg_file:
            st.error("뇌파 파일을 업로드해주세요.")
            return
        
        if not interview_file:
            st.error("인터뷰 파일을 업로드해주세요.")
            return
        
        if not manual_id or not manual_id.strip():
            st.error("참가자 ID를 입력해주세요.")
            return
        
        participant_id = manual_id.strip()
        
        # 파일 저장
        with st.spinner("파일 저장 중..."):
            # 뇌파 파일 저장
            eeg_dir = project_root / "data" / "Emotion_EEG" / "VR_Result_Data"
            eeg_filename = f"RECORD_{participant_id}.txt"
            eeg_path = save_uploaded_file(eeg_file, eeg_dir / eeg_filename)
            
            # 인터뷰 파일 저장
            interview_dir = project_root / "data" / "txt_files"
            interview_filename = f"{participant_id}.txt"
            interview_path = save_uploaded_file(interview_file, interview_dir / interview_filename)
            
            st.success(f"파일 저장 완료!")
            st.info(f"• 뇌파 파일: {eeg_path}")
            st.info(f"• 인터뷰 파일: {interview_path}")
        
        # 분석 실행
        st.markdown("---")
        st.markdown("### 분석 진행 중...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 분석 실행
        with st.spinner("분석을 실행하고 있습니다. 잠시만 기다려주세요..."):
            success, output, error = run_analysis(participant_id)
        
        progress_bar.progress(100)
        
        # 출력 로그 확인 - 실제 오류만 감지
        has_critical_errors = False
        if output:
            # 실제 오류 메시지만 확인 (경고는 제외)
            critical_error_patterns = [
                "Traceback",
                "Error code:",
                "PermissionDenied",
                "ModuleNotFoundError",
                "FileNotFoundError",
                "실패했습니다",
                "실행 실패",
                "❌",
                "🛑"
            ]
            # PEFT 관련 경고는 정상이므로 제외
            output_without_peft = output.replace("PEFT 모듈이 없어", "").replace("PEFT", "")
            has_critical_errors = any(pattern in output_without_peft for pattern in critical_error_patterns)
        
        if error and "Traceback" in str(error):
            has_critical_errors = True
        
        # 결과 파일 생성 확인
        result_files_exist = False
        expected_files = [
            f"output/emotionRelation/visualization/{participant_id}_connected_group.html",
            f"output/Emotion_EEG/Report_Json_Data/Report_Data.json",
            f"output/llama3/{participant_id}_llama_analysis.json"
        ]
        
        existing_files = [f for f in expected_files if Path(f).exists()]
        if len(existing_files) > 0:
            result_files_exist = True
        
        # "모든 작업 완료!" 메시지가 있으면 성공으로 간주
        analysis_completed = "모든 작업 완료!" in output if output else False
        
        if success and (not has_critical_errors or analysis_completed) and result_files_exist:
            st.markdown("""
            <div class="success-box">
                <strong>분석 완료!</strong><br>
                모든 분석 단계가 성공적으로 완료되었습니다.
            </div>
            """, unsafe_allow_html=True)
            
            # 출력 로그 표시 (선택적)
            with st.expander("분석 로그 보기", expanded=False):
                if output:
                    st.text_area("출력 로그", output, height=300)
                if error:
                    st.text_area("오류 로그", error, height=200)
            
            # 결과 보기 버튼
            st.markdown("---")
            st.markdown("### 분석 결과 확인")
            st.success(f"참가자 **{participant_id}**의 분석이 완료되었습니다!")
            st.info("왼쪽 사이드바의 '대시보드' 메뉴에서 결과를 확인하실 수 있습니다.")
            
            # 세션 상태에 participant_id 저장
            st.session_state['participant_id'] = participant_id
            st.session_state['participant_name'] = participant_id  # 이름도 저장
            
            # 자동으로 대시보드로 이동하는 옵션
            if st.button("대시보드에서 결과 보기", use_container_width=True):
                # 세션 상태를 확실히 저장한 후 페이지 이동
                st.session_state['participant_id'] = participant_id
                st.session_state['participant_name'] = participant_id
                # Streamlit의 switch_page는 페이지 이름만 사용 (파일명에서 숫자와 언더스코어 제거)
                st.switch_page("대시보드")
        elif success and has_critical_errors and not analysis_completed:
            # 실행은 되었지만 오류가 있거나 결과 파일이 없는 경우
            st.markdown("""
            <div class="error-box">
                <strong>분석 실행 중 문제가 발생했습니다</strong><br>
                출력 로그를 확인하여 문제를 파악하세요.
            </div>
            """, unsafe_allow_html=True)
            
            # 출력 로그 표시 (필수)
            with st.expander("분석 로그 보기 (필수 확인)", expanded=True):
                if output:
                    st.text_area("출력 로그", output, height=400)
                if error:
                    st.text_area("오류 로그", error, height=200)
            
            if not result_files_exist:
                st.warning(f"⚠️ 결과 파일이 생성되지 않았습니다. 예상 경로: {expected_files[0]}")
            
            st.info("출력 로그를 확인하여 문제를 해결한 후 다시 시도해주세요.")
        else:
            st.markdown(f"""
            <div class="error-box">
                <strong>분석 실패</strong><br>
                분석 중 오류가 발생했습니다: {output}
            </div>
            """, unsafe_allow_html=True)
            
            if error:
                with st.expander("오류 상세 정보", expanded=True):
                    st.text(error)

if __name__ == "__main__":
    main()


import sys
import os
import glob
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def step1_main_emotion_extraction(participant_id: str):
    print("\n" + "="*60)
    print("1-1. 메인 감정 추출 (Step1)")
    print("="*60)
    
    try:
        import importlib.util
        step1_path = current_dir / "emotion-interview_relation" / "step1_extract_main_emotion.py"
        spec = importlib.util.spec_from_file_location("step1", step1_path)
        step1_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step1_module)
        step1_module.process_single_file(participant_id)
        print("✅ Step1 완료")
    except Exception as e:
        print(f"❌ Step1 오류: {e}")
        import traceback
        traceback.print_exc()

def step2_keyword_extraction(participant_id: str):
    print("\n" + "="*60)
    print("1-2. 키워드 추출 (Step2)")
    print("="*60)
    
    try:
        import importlib.util
        step2_path = current_dir / "emotion-interview_relation" / "step2_keyword_extractor.py"
        spec = importlib.util.spec_from_file_location("step2", step2_path)
        step2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step2_module)
        
        extractor = step2_module.KeywordExtractor()
        txt_file = f"{participant_id}.txt"
        txt_path = Path(f"data/txt_files/{txt_file}")
        
        if txt_path.exists():
            extractor.analyze_single_file(txt_file, participant_id)
            print("✅ Step2 완료")
        else:
            print(f"⚠️ {txt_path} 파일이 없습니다. 건너뜁니다.")
    except Exception as e:
        print(f"❌ Step2 오류: {e}")
        import traceback
        traceback.print_exc()

def step3_relation_analysis(participant_id: str):
    print("\n" + "="*60)
    print("1-3. 연관성 분석 (Step3)")
    print("="*60)
    
    try:
        import importlib.util
        step3_path = current_dir / "emotion-interview_relation" / "step3_relation_analyzer.py"
        spec = importlib.util.spec_from_file_location("step3", step3_path)
        step3_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step3_module)
        
        analyzer = step3_module.RelationAnalyzer()
        analyzer.analyze_single_file(participant_id)
        print("✅ Step3 완료")
    except Exception as e:
        print(f"❌ Step3 오류: {e}")
        import traceback
        traceback.print_exc()

def step4_visualization(participant_id: str):
    print("\n" + "="*60)
    print("1-4. 시각화 (Step4)")
    print("="*60)
    
    try:
        import importlib.util
        step4_path = current_dir / "emotion-interview_relation" / "step4_visualizer.py"
        spec = importlib.util.spec_from_file_location("step4", step4_path)
        step4_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step4_module)
        
        visualizer = step4_module.NetworkVisualizerPNG()
        visualizer.visualize_single_file(f"{participant_id}_finalRelation.json")
        print("✅ Step4 완료")
    except Exception as e:
        print(f"❌ Step4 오류: {e}")
        import traceback
        traceback.print_exc()

def step1_self_awareness_graph(participant_id: str):
    print("\n" + "="*60)
    print("1. VR 감정 정보 기반 자기 인식 및 해석 그래프")
    print("="*60)
    
    step1_main_emotion_extraction(participant_id)
    step2_keyword_extraction(participant_id)
    step3_relation_analysis(participant_id)
    step4_visualization(participant_id)
    
    print("\n✅ 1단계 완료")

def step2_eeg_indicators_visualization(participant_id: str):
    print("\n" + "="*60)
    print("2. 뇌파 데이터 6가지 지표 시각화")
    print("="*60)
    
    try:
        from src.Emotion_EEG.EEG_Color.EEG_Table_Visualizer import main as table_main
        
        table_main()
        print("✅ 2단계 완료")
    except Exception as e:
        print(f"❌ 2단계 오류: {e}")
        import traceback
        traceback.print_exc()

def step3_radar_chart(participant_id: str):
    print("\n" + "="*60)
    print("3. Radar Chart 출력")
    print("="*60)
    
    try:
        from src.Emotion_EEG.Rader_Chart.RaderChart import run_rader_chart
        
        run_rader_chart()
        print("✅ 3단계 완료")
    except Exception as e:
        print(f"❌ 3단계 오류: {e}")
        import traceback
        traceback.print_exc()

def step4_eeg_summary(participant_id: str):
    print("\n" + "="*60)
    print("4. 뇌파 데이터를 바탕으로 요약 서술")
    print("="*60)
    
    try:
        print("\n4-1. JSON을 JSONL로 변환 중...")
        from src.Emotion_EEG.JsonToJsonl.JsonToJsonlMain import run_json_to_jsonl
        
        jsonl_success = run_json_to_jsonl()
        if not jsonl_success:
            print("⚠️ JSONL 변환 실패. 4단계를 건너뜁니다.")
            return
        
        print("\n4-2. Llama3 모델로 요약 생성 중...")
        from src.Emotion_EEG.DescriptiveSummary_Llama3.Llama3Main import run_llama_inference
        
        inference_success = run_llama_inference()
        if inference_success:
            print("✅ 4단계 완료")
        else:
            print("⚠️ Llama3 추론 실패. 모델 어댑터가 없거나 GPU 환경 문제일 수 있습니다.")
    except Exception as e:
        print(f"❌ 4단계 오류: {e}")
        import traceback
        traceback.print_exc()

def step5_emotion_frequency_wordcloud(participant_id: str):
    print("\n" + "="*60)
    print("5. 감정 빈도 분석 그래프 및 워드 클라우드")
    print("="*60)
    
    try:
        import importlib.util
        step6_path = current_dir / "vr-interview" / "step6_virtualization_emotionWord.py"
        spec = importlib.util.spec_from_file_location("step6", step6_path)
        step6_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step6_module)
        
        visualizer = step6_module.EmotionVisualizer()
        visualizer.visualize_single_file(participant_id)
        print("✅ 5단계 완료")
    except Exception as e:
        print(f"❌ 5단계 오류: {e}")
        import traceback
        traceback.print_exc()

def step6_topic_network(participant_id: str):
    print("\n" + "="*60)
    print("6. 토픽 네트워크맵")
    print("="*60)
    
    try:
        import json
        import networkx as nx
        from sentence_transformers import SentenceTransformer, util
        import importlib.util
        graph_renderer_path = current_dir / "graph_renderer.py"
        spec_renderer = importlib.util.spec_from_file_location("graph_renderer", graph_renderer_path)
        graph_renderer_module = importlib.util.module_from_spec(spec_renderer)
        spec_renderer.loader.exec_module(graph_renderer_module)
        render_graph = graph_renderer_module.render_graph
        
        target_file = f"output/llama3/{participant_id}_llama_analysis.json"
        output_html = f"{participant_id}_graph_white_static.html"
        model_name = "jhgan/ko-sbert-multitask"
        
        if not Path(target_file).exists():
            print(f"⚠️ {target_file} 파일이 없습니다. 건너뜁니다.")
            return
        
        print(f"SBERT 모델 로딩 중({model_name})")
        embedder = SentenceTransformer(model_name)
        
        with open(target_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, dict) and "analysis_result" in raw_data:
            data_nodes = raw_data["analysis_result"]
        elif isinstance(raw_data, list):
            data_nodes = raw_data
        else:
            data_nodes = []
        
        if not data_nodes:
            print("⚠️ 데이터가 없습니다.")
            return
        
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
        render_graph(nx_graph, data_nodes, output_html)
        print(f"✅ 그래프 생성 완료: {Path(output_html).absolute()}")
        print("✅ 6단계 완료")
    except Exception as e:
        print(f"❌ 6단계 오류: {e}")
        import traceback
        traceback.print_exc()

def step7_eeg_interview_consistency(participant_id: str):
    print("\n" + "="*60)
    print("7. 뇌파 - 인터뷰 일치")
    print("="*60)
    
    try:
        from src.module.EmoriAnalyzer import EmoriAnalyzer
        from src.module.EmoriVisualizer import EmoriVisualizer
        
        eeg_path = "output/Emotion_EEG/Report_Json_Data/Report_Data.json"
        llama_path = f"output/llama3/{participant_id}_llama_analysis.json"
        
        if not Path(eeg_path).exists() or not Path(llama_path).exists():
            print(f"⚠️ 필요한 파일이 없습니다. 건너뜁니다.")
            return
        
        analyzer = EmoriAnalyzer(eeg_path, llama_path)
        result = analyzer.analyze()
        
        if result:
            visualizer = EmoriVisualizer(output_dir="output/report_images")
            visualizer.plot_discrepancy(
                stress_score=result['discrepancy']['stress_val'],
                text_score=result['discrepancy']['text_val'],
                filename=f"{participant_id}_discrepancy.png"
            )
            print(f"✅ 일치 분석 완료")
        
        print("✅ 7단계 완료")
    except Exception as e:
        print(f"❌ 7단계 오류: {e}")
        import traceback
        traceback.print_exc()

def step8_final_report(participant_id: str):
    print("\n" + "="*60)
    print("8. 최종 요약 서술")
    print("="*60)
    
    try:
        from src.FinalReportGenerator import FinalReportGenerator
        
        generator = FinalReportGenerator(base_dir="output", participant_id=participant_id)
        report = generator.generate()
        
        output_path = Path("output") / "final_reports" / f"{participant_id}_final_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"✅ 최종 보고서 저장 완료: {output_path}")
        print("✅ 8단계 완료")
    except Exception as e:
        print(f"❌ 8단계 오류: {e}")
        import traceback
        traceback.print_exc()

def preprocess_eeg_data():
    print("\n" + "="*60)
    print("전처리: EEG 데이터 변환")
    print("="*60)
    
    try:
        from src.Emotion_EEG.TxtToJson.TxtToJson import run_txt_to_json
        
        success = run_txt_to_json()
        if success:
            print("✅ EEG 데이터 변환 완료")
        else:
            print("⚠️ EEG 데이터 변환 실패 또는 파일 없음")
    except Exception as e:
        print(f"❌ EEG 데이터 변환 오류: {e}")
        import traceback
        traceback.print_exc()

def preprocess_interview_doheon(participant_id: str):
    print("\n" + "="*60)
    print("전처리: 인터뷰 분석 (도현 방식 - 6번용)")
    print("="*60)
    
    try:
        import importlib.util
        extract_path = current_dir / "extract_emotion_json.py"
        spec = importlib.util.spec_from_file_location("extract_emotion", extract_path)
        extract_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extract_module)
        
        from dotenv import load_dotenv
        from groq import Groq
        import os
        import json
        
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY가 설정되지 않았습니다. 건너뜁니다.")
            return
        
        client = Groq(api_key=api_key)
        txt_file = f"{participant_id}.txt"
        
        if Path(f"data/txt_files/{txt_file}").exists():
            result = extract_module.analyze_file(client, txt_file)
            if result:
                output_filename = f"{participant_id}_llama_analysis.json"
                output_path = Path("output/llama3") / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 인터뷰 분석 완료: {output_path}")
            else:
                print("⚠️ 인터뷰 분석 결과가 없습니다.")
        else:
            print(f"⚠️ data/txt_files/{txt_file} 파일이 없습니다.")
    except Exception as e:
        print(f"❌ 인터뷰 분석 오류: {e}")
        import traceback
        traceback.print_exc()

def preprocess_interview_hyunwoo(participant_id: str):
    print("\n" + "="*60)
    print("전처리: 인터뷰 분석 (현우 방식 - 5번용)")
    print("="*60)
    
    try:
        import importlib.util
        
        step2_path = current_dir / "vr-interview" / "step2_morpheme_analysis.py"
        spec2 = importlib.util.spec_from_file_location("step2_morpheme", step2_path)
        step2_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(step2_module)
        
        step4_path = current_dir / "vr-interview" / "step4_keyword_extraction.py"
        spec4 = importlib.util.spec_from_file_location("step4_keyword", step4_path)
        step4_module = importlib.util.module_from_spec(spec4)
        spec4.loader.exec_module(step4_module)
        
        txt_file = f"{participant_id}.txt"
        txt_path = Path(f"data/txt_files/{txt_file}")
        
        if not txt_path.exists():
            print(f"⚠️ data/txt_files/{txt_file} 파일이 없습니다.")
            return
        
        print("2-1. 형태소 분석 중...")
        morpheme_analyzer = step2_module.MorphemeAnalyzer()
        morpheme_analyzer.analyze_file(str(txt_path))
        
        print("2-2. 키워드 추출 중...")
        morpheme_filename = f"{participant_id}_morpheme.json"
        keyword_analyzer = step4_module.LlamaSbertAnalyzer()
        keyword_analyzer.analyze_single_file(morpheme_filename)
        
        print("✅ 인터뷰 분석 완료")
    except Exception as e:
        print(f"❌ 인터뷰 분석 오류: {e}")
        import traceback
        traceback.print_exc()

def main(participant_id: str):
    print("="*60)
    print("Emori 프로젝트 통합 실행")
    print(f"참가자 ID: {participant_id}")
    print("="*60)
    
    print("\n[전처리 단계]")
    preprocess_eeg_data()
    preprocess_interview_doheon(participant_id)
    preprocess_interview_hyunwoo(participant_id)
    
    print("\n[실제 출력 단계]")
    step1_self_awareness_graph(participant_id)
    step2_eeg_indicators_visualization(participant_id)
    step3_radar_chart(participant_id)
    step4_eeg_summary(participant_id)
    step5_emotion_frequency_wordcloud(participant_id)
    step6_topic_network(participant_id)
    step7_eeg_interview_consistency(participant_id)
    step8_final_report(participant_id)
    
    print("\n" + "="*60)
    print("모든 작업 완료!")
    print("="*60)

if __name__ == "__main__":
    import re
    
    if len(sys.argv) < 2:
        print("사용법: python src/main.py <참가자_ID>")
        print("예시: python src/main.py EB_002")
        print("또는: python src/main.py data/txt_files/EB_002.txt")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg.endswith('.txt'):
        path = Path(arg)
        participant_id = path.stem
        print(f"파일 경로에서 참가자 ID 추출: {participant_id}")
    elif '/' in arg or '\\' in arg:
        path = Path(arg)
        participant_id = path.stem
        print(f"경로에서 참가자 ID 추출: {participant_id}")
    else:
        participant_id = arg
    
    if not re.match(r'^[A-Z]{2}_\d{3}$', participant_id):
        print(f"⚠️ 경고: '{participant_id}'가 일반적인 참가자 ID 형식(예: EB_001)이 아닙니다.")
        print(f"계속 진행합니다...")
    
    main(participant_id)

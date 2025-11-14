"""
1단계: HWP 파일을 TXT 파일로 변환 (olefile 전문 버전)
HWP 5.0 파일 구조를 완전히 분석하여 텍스트 추출
"""

import os
import struct
import zlib
from pathlib import Path
import olefile

class HWPParser:
    """HWP 5.0 파일 구조 파서"""
    
    HWPTAG_BEGIN = 0x10
    HWPTAG_TEXT = 67
    
    def __init__(self, hwp_path):
        self.hwp_path = hwp_path
        self.ole = None
        
    def open(self):
        """HWP 파일 열기"""
        try:
            self.ole = olefile.OleFileIO(self.hwp_path)
            return True
        except Exception as e:
            print(f"  ❌ 파일 열기 실패: {e}")
            return False
    
    def close(self):
        """HWP 파일 닫기"""
        if self.ole:
            self.ole.close()
    
    def decompress_stream(self, data):
        """압축된 스트림 해제"""
        try:
            decompressed = zlib.decompress(data, -zlib.MAX_WBITS)
            return decompressed
        except:
            return data
    
    def extract_text_from_section(self, section_data):
        """섹션 데이터에서 텍스트 추출"""
        data = self.decompress_stream(section_data)
        texts = []
        
        # 방법 1: HWP 레코드 구조 파싱
        pos = 0
        while pos < len(data) - 4:
            try:
                header = struct.unpack('<I', data[pos:pos+4])[0]
                tag_id = header & 0x3FF
                size = (header >> 20) & 0xFFF
                pos += 4
                
                if size == 0xFFF:
                    if pos + 4 <= len(data):
                        size = struct.unpack('<I', data[pos:pos+4])[0]
                        pos += 4
                
                if tag_id == self.HWPTAG_TEXT or tag_id == 67:
                    if pos + size <= len(data):
                        text_data = data[pos:pos+size]
                        try:
                            text = text_data.decode('utf-16le', errors='ignore')
                            text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t ')
                            if text.strip():
                                texts.append(text.strip())
                        except:
                            pass
                
                pos += size
            except:
                pos += 1
                continue
        
        # 방법 2: UTF-16LE 디코딩
        if not texts:
            try:
                decoded = data.decode('utf-16le', errors='ignore')
                cleaned = ''.join(c for c in decoded if c.isprintable() or c in '\n\r\t ')
                if cleaned.strip():
                    texts.append(cleaned.strip())
            except:
                pass
        
        # 방법 3: 바이트 패턴 검색
        if not texts:
            text_parts = []
            i = 0
            while i < len(data) - 1:
                try:
                    char_code = struct.unpack('<H', data[i:i+2])[0]
                    if (0xAC00 <= char_code <= 0xD7A3) or \
                       (0x0020 <= char_code <= 0x007E) or \
                       (0x3000 <= char_code <= 0x9FFF):
                        try:
                            char = chr(char_code)
                            text_parts.append(char)
                        except:
                            pass
                    elif char_code == 0x000D:
                        text_parts.append('\n')
                    i += 2
                except:
                    i += 1
            
            if text_parts:
                text = ''.join(text_parts)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if lines:
                    texts.extend(lines)
        
        return '\n'.join(texts)
    
    def extract_all_text(self):
        """HWP 파일에서 모든 텍스트 추출"""
        if not self.ole:
            return None
        
        all_texts = []
        section_num = 0
        
        while True:
            section_name = f'BodyText/Section{section_num}'
            if not self.ole.exists(section_name):
                break
            
            try:
                stream = self.ole.openstream(section_name)
                section_data = stream.read()
                text = self.extract_text_from_section(section_data)
                
                if text and text.strip():
                    all_texts.append(text.strip())
                    print(f"  ✓ Section{section_num}: {len(text)} 문자 추출")
            except Exception as e:
                print(f"  ⚠️  Section{section_num} 오류: {e}")
            
            section_num += 1
        
        return '\n\n'.join(all_texts) if all_texts else None


class HWPConverter:
    def __init__(self, input_folder="data/hwp_files", output_folder="data/txt_files"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
    
    def convert_single_file(self, hwp_filename):
        """단일 HWP 파일을 TXT로 변환"""
        hwp_path = os.path.join(self.input_folder, hwp_filename)
        
        if not os.path.exists(hwp_path):
            print(f"❌ 파일을 찾을 수 없습니다: {hwp_path}")
            return False
        
        print(f"\n{'='*60}")
        print(f" 변환 중: {hwp_filename}")
        print('='*60)
        
        parser = HWPParser(hwp_path)
        if not parser.open():
            return False
        
        text = parser.extract_all_text()
        parser.close()
        
        if not text or len(text.strip()) < 10:
            print(f"\n❌ 변환 실패: 유효한 텍스트를 찾을 수 없습니다")
            return False
        
        korean_count = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
        
        txt_filename = Path(hwp_filename).stem + '.txt'
        txt_path = os.path.join(self.output_folder, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"\n✅ 변환 완료!")
        print(f"    저장 위치: {txt_path}")
        print(f"    텍스트 길이: {len(text)} 문자")
        print(f"    한글 문자: {korean_count}개")
        
        preview_lines = []
        for line in text.split('\n')[:8]:
            if line.strip():
                preview_lines.append(line.strip()[:70])
                if len(preview_lines) >= 5:
                    break
        
        if preview_lines:
            print(f"\n    미리보기:")
            print("   " + "-"*56)
            for line in preview_lines:
                print(f"   {line}")
            print("   " + "-"*56)
        
        if korean_count > 50:
            print(f"\n   정상적으로 변환되었습니다!")
            return True
        elif korean_count > 10:
            print(f"\n   ⚠️  일부만 변환되었을 수 있습니다.")
            return True
        else:
            print(f"\n   ❌ 한글이 거의 없습니다.")
            return False
    
    def convert_all_files(self):
        """모든 HWP 파일 변환"""
        hwp_files = sorted([f for f in os.listdir(self.input_folder) if f.endswith('.hwp')])
        
        if not hwp_files:
            print(f"❌ HWP 파일을 찾을 수 없습니다: {self.input_folder}")
            return 0
        
        print(f"\n 총 {len(hwp_files)}개의 HWP 파일을 찾았습니다.")
        
        success_count = 0
        failed_files = []
        
        for i, hwp_file in enumerate(hwp_files, 1):
            print(f"\n[{i}/{len(hwp_files)}]", end=" ")
            if self.convert_single_file(hwp_file):
                success_count += 1
            else:
                failed_files.append(hwp_file)
        
        print(f"\n\n{'='*60}")
        print(f" 변환 결과 요약")
        print('='*60)
        print(f"✅ 성공: {success_count}/{len(hwp_files)} 파일")
        
        if failed_files:
            print(f"\n❌ 실패한 파일 ({len(failed_files)}개):")
            for f in failed_files:
                print(f"   - {f}")
        
        print('='*60)
        return success_count


def main():
    print("\n" + "="*60)
    print(" HWP to TXT 변환 프로그램 (olefile)")
    print("="*60)
    
    converter = HWPConverter(
        input_folder="data/hwp_files",
        output_folder="data/txt_files"
    )
    
    print("\n변환 모드를 선택하세요:")
    print("1. 모든 HWP 파일 변환")
    print("2. 특정 파일만 변환")
    print("3. 종료")
    
    choice = input("\n선택 (1, 2, 3): ").strip()
    
    if choice == '1':
        converter.convert_all_files()
    elif choice == '2':
        filename = input("\nHWP 파일명 입력 (예: EG_001.hwp): ").strip()
        converter.convert_single_file(filename)
    elif choice == '3':
        print("프로그램을 종료합니다.")
    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()
import json
import re
from statistics import mean
# ---------- 값 추출 by 정규표현식(라벨 : 값) ----------
def extract_line_value(label, text):
    m = re.search(rf"^{re.escape(label)}\s*:\s*([^\n])$", text, flags=re.MULTILINE)
    if m:
        extracted_value = m.group(1).strip()
        # 값이 공백이거나 비어있으면 'NULL' 반환
        return extracted_value if extracted_value else "NULL"
    else:
        # 라벨 자체를 찾지 못했으면 'NULL' 반환
        return "NULL"
# ---------- UserInfo ----------
name = extract_line_value("NAME", raw)
age = extract_line_value("AGE", raw)
gender = extract_line_value("GENDER", raw)

# ---------- OwnerInfo ----------
date = extract_line_value("REPORT_DAY", raw)

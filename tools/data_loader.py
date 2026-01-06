import json
import os
import re
from typing import List, Tuple
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Các tiêu đề mục thường gặp trong Gale Encyclopedia of Medicine
GALE_SECTION_PATTERNS = [
    r'^(Definition)\s*$',
    r'^(Description)\s*$',
    r'^(Causes and symptoms)\s*$',
    r'^(Causes)\s*$',
    r'^(Symptoms)\s*$',
    r'^(Diagnosis)\s*$',
    r'^(Treatment)\s*$',
    r'^(Prognosis)\s*$',
    r'^(Prevention)\s*$',
    r'^(Aftercare)\s*$',
    r'^(Risks)\s*$',
    r'^(Preparation)\s*$',
    r'^(Purpose)\s*$',
    r'^(Precautions)\s*$',
    r'^(Side effects)\s*$',
    r'^(Interactions)\s*$',
    r'^(Resources)\s*$',
    r'^(KEY TERMS)\s*$',
]


def _extract_entry_title(text: str) -> str:
    """
    Trích xuất tên entry (tên bệnh) từ nội dung trang PDF.
    Thông thường nằm ở các dòng đầu và không trùng với tiêu đề mục y khoa.
    """
    lines = text.split('\n')

    for line in lines[:10]:
        line = line.strip()

        if 3 <= len(line) <= 60 and line[0].isupper():
            if not any(re.match(p, line, re.IGNORECASE) for p in GALE_SECTION_PATTERNS):
                if not re.match(r'^[A-Z\s]{10,}$', line):
                    return line

    return ""


def _parse_sections(text: str, page_num: int) -> List[Tuple[str, str, int]]:
    """
    Tách nội dung trang thành các mục y khoa dựa trên tiêu đề.
    Mỗi mục trả về: (tên mục, nội dung, số trang)
    """
    sections = []

    combined_pattern = '|'.join(f'({p})' for p in GALE_SECTION_PATTERNS)
    matches = list(re.finditer(combined_pattern, text, re.MULTILINE | re.IGNORECASE))

    if not matches:
        return [("Content", text.strip(), page_num)]

    for i, match in enumerate(matches):
        section_name = match.group().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[start:end].strip()

        if section_name.upper() in {"RESOURCES", "KEY TERMS"}:
            continue

        if content and len(content) > 50:
            sections.append((section_name, content, page_num))

    return sections


def _extract_text_with_columns(page) -> str:
    """
    Trích xuất văn bản từ PDF có bố cục 2 cột.
    Đọc cột trái từ trên xuống dưới, sau đó đến cột phải.
    """
    blocks = page.get_text("blocks", sort=True)
    mid_x = page.rect.width / 2

    left_col = []
    right_col = []

    for b in blocks:
        if len(b) >= 5 and b[4]:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            center_x = (x0 + x1) / 2

            if center_x < mid_x:
                left_col.append((y0, text.strip()))
            else:
                right_col.append((y0, text.strip()))

    left_col.sort(key=lambda x: x[0])
    right_col.sort(key=lambda x: x[0])

    return (
        '\n'.join(t[1] for t in left_col)
        + '\n'
        + '\n'.join(t[1] for t in right_col)
    )


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Đọc PDF y khoa (Gale Encyclopedia), tách theo từng mục y khoa
    và trả về danh sách Document cho RAG.
    """
    if not os.path.exists(pdf_path):
        print(f"Không tìm thấy file PDF: {pdf_path}")
        return []

    docs = []
    current_entry = ""

    try:
        pdf_doc = fitz.open(pdf_path)

        for page_index, page in enumerate(pdf_doc):
            text = _extract_text_with_columns(page)

            if len(text.strip()) < 100:
                continue

            title = _extract_entry_title(text)
            if title:
                current_entry = title

            sections = _parse_sections(text, page_index + 1)

            for section_name, section_content, page_num in sections:
                if current_entry:
                    content = (
                        f"Entry: {current_entry}\n\n"
                        f"{section_name}: {section_content}"
                    )
                else:
                    content = f"{section_name}: {section_content}"

                metadata = {
                    "source": "Gale Encyclopedia of Medicine",
                    "entry": current_entry or "Unknown",
                    "section": section_name,
                    "page": page_num,
                }

                docs.append(Document(page_content=content, metadata=metadata))

        pdf_doc.close()
        return docs

    except Exception as e:
        print(f"Lỗi khi đọc PDF: {e}")
        return []


def load_json(json_path: str) -> List[Document]:
    """
    Đọc dữ liệu y khoa từ JSON.
    Mỗi mục thông tin của bệnh được tách thành một chunk riêng.
    """
    if not os.path.exists(json_path):
        print(f"Không tìm thấy file JSON: {json_path}")
        return []

    docs = []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            disease_name = entry.get('ten_benh', 'Unknown Disease')
            source_url = entry.get('url_nguon', 'Medical JSON Database')

            for key, value in entry.items():
                if key in {'ten_benh', 'url_nguon'} or not isinstance(value, str):
                    continue

                if not value.strip():
                    continue

                content = f"Bệnh: {disease_name}\n\n{key}: {value}"

                metadata = {
                    "source": source_url,
                    "disease": disease_name,
                    "section": key,
                }

                docs.append(Document(page_content=content, metadata=metadata))

        return docs

    except Exception as e:
        print(f"Lỗi khi đọc JSON: {e}")
        return []


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Chia nhỏ các chunk lớn để phù hợp với embedding model.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", ". ", "\n", " "]
    )
    return splitter.split_documents(docs)


def process_data(pdf_path: str = None, json_path: str = None) -> List[Document]:
    """
    Pipeline tổng hợp: đọc PDF / JSON, chunk hóa và trả về Document list.
    """
    all_docs = []

    if pdf_path:
        all_docs.extend(load_pdf(pdf_path))

    if json_path:
        all_docs.extend(load_json(json_path))

    if not all_docs:
        return []

    large_docs = [d for d in all_docs if len(d.page_content) > 2000]
    small_docs = [d for d in all_docs if len(d.page_content) <= 2000]

    if large_docs:
        large_docs = split_documents(large_docs)

    return small_docs + large_docs

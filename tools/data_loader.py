import json
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_path: str) -> List[Document]:
    """Load documents from a PDF file"""
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at {pdf_path}")
        return []

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages from PDF")
        return docs
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []


def load_json(json_path: str) -> List[Document]:
    """Load documents from a JSON file"""
    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}")
        return []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        docs = []
        for entry in data:
            # Construct content from all fields
            content_parts = []

            # Add disease name as header
            if 'ten_benh' in entry:
                content_parts.append(f"Bệnh: {entry['ten_benh']}")

            # Add other fields
            for key, value in entry.items():
                if key not in ['ten_benh', 'url_nguon'] and isinstance(value, str):
                    content_parts.append(f"{key}: {value}")

            text_content = "\n\n".join(content_parts)

            # Create metadata
            metadata = {
                "source": entry.get('url_nguon', 'Medical JSON Database'),
                "title": entry.get('ten_benh', 'Unknown Disease')
            }

            docs.append(Document(page_content=text_content, metadata=metadata))

        print(f"Loaded {len(docs)} entries from JSON")
        return docs
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", ". ", "\n", " "]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks")
    return splits


def process_data(pdf_path: str = None, json_path: str = None) -> List[Document]:
    """Load and process data from available sources"""
    all_docs = []

    if pdf_path:
        print(f"Processing PDF: {pdf_path}")
        all_docs.extend(load_pdf(pdf_path))

    if json_path:
        print(f"Processing JSON: {json_path}")
        all_docs.extend(load_json(json_path))

    if not all_docs:
        print("No documents loaded from any source")
        return []

    return split_documents(all_docs)

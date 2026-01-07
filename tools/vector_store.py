import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Singleton cho embeddings và vectorstore
_embeddings = None
_vectorstore = None


def get_embeddings():
    """
    Khởi tạo và trả về embedding model.
    """
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )

    return _embeddings


def get_or_create_vectorstore(documents=None, persist_dir="./medical_db/"):
    """
    Load vector database nếu đã tồn tại.
    Nếu chưa có và được truyền documents thì tạo mới.
    """
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    db_files_exist = False
    files = os.listdir(persist_dir)
    db_files_exist = any(
        f.endswith(".sqlite3") or f == "chroma.sqlite3" or f.startswith("index")
        for f in files
    )

    if db_files_exist:
        print("Đang load vector database đã tồn tại...")
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

        collection = _vectorstore._collection
        if collection.count() == 0:
            print("Vector database rỗng, cần tạo lại")
            _vectorstore = None
            return None

        print(f"Đã load {collection.count()} document từ vector database")

    elif documents:
        print("Đang tạo vector database mới...")
        _vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )

        print(f"Đã tạo vector database với {len(documents)} document")

    else:
        print("Không có vector database và cũng không có document để tạo mới")
        return None

    return _vectorstore


def get_retriever(k=3):
    """
    Trả về retriever từ vectorstore hiện có.
    """
    vectorstore = get_or_create_vectorstore()

    if vectorstore:
        return vectorstore.as_retriever(search_kwargs={"k": k})

    return None

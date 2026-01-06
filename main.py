from dotenv import load_dotenv

from core.langgraph_workflow import create_workflow
from core.state import initialize_conversation_state, reset_query_state
from tools.data_loader import process_data
from tools.vector_store import get_or_create_vectorstore

load_dotenv()


def initialize_system():
    """
    Khởi tạo hệ thống và vector database (nếu chưa tồn tại).
    """
    pdf_path = "./data/medical_book.pdf"
    json_path = "./data/medical-data.json"
    persist_dir = "./medical_db/"

    print("\n" + "=" * 60)
    print("Initializing Medical AI System...")
    print("=" * 60)

    # Thử load vector database đã tồn tại
    existing_db = get_or_create_vectorstore(persist_dir=persist_dir)

    if not existing_db:
        print("Đang xử lý dữ liệu và tạo vector database...")
        doc_splits = process_data(pdf_path=pdf_path, json_path=json_path)

        if doc_splits:
            vectorstore = get_or_create_vectorstore(
                documents=doc_splits,
                persist_dir=persist_dir
            )
            if vectorstore:
                print("Tạo vector database thành công")
            else:
                print("Không thể tạo vector database")
        else:
            print("Không có dữ liệu để tạo vector database")
            print("Hệ thống sẽ chạy với chức năng hạn chế (không dùng RAG)")


def main():
    # Khởi tạo hệ thống
    initialize_system()

    print("\nCreating workflow...")
    app = create_workflow()

    conversation_state = initialize_conversation_state()

    print("\n" + "=" * 60)
    print("Medical AI Assistant Ready!")
    print("=" * 60)
    print("Lệnh: 'exit' để thoát, 'clear' để xoá hội thoại")
    print("Hãy đặt câu hỏi về y tế, sức khỏe\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() == "exit":
            print("\nCảm ơn bạn đã sử dụng Medical AI Assistant. Chúc bạn nhiều sức khỏe!")
            break

        if query.lower() == "clear":
            conversation_state = initialize_conversation_state()
            print("\nĐã xoá hội thoại. Bắt đầu lại!\n")
            continue

        if not query:
            print("Vui lòng nhập câu hỏi.\n")
            continue

        # Reset trạng thái cho câu hỏi mới nhưng giữ lịch sử hội thoại
        conversation_state = reset_query_state(conversation_state)
        conversation_state["question"] = query

        print("\nĐang xử lý câu hỏi...")

        result = app.invoke(conversation_state)
        conversation_state.update(result)

        if result.get("generation"):
            print(f"\nResponse: {result['generation']}")
            print(f"Source: {result.get('source', 'Unknown')}")
        else:
            print("\nKhông thể tạo câu trả lời. Vui lòng thử lại.")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()

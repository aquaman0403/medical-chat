# Medical Chat API

Medical Chat API là một dự án chatbot y tế thông minh được xây dựng bằng Flask và LangChain. Hệ thống sử dụng kiến trúc multi-agent mạnh mẽ dựa trên LangGraph để trả lời các câu hỏi về y khoa một cách chính xác và linh hoạt.

## Tổng quan

Dự án này triển khai một API backend cho ứng dụng chatbot y tế. Chatbot có khả năng:
-   Hiểu và phân tích câu hỏi của người dùng.
-   Lập kế hoạch và quyết định cách tốt nhất để tìm câu trả lời.
-   Truy vấn cơ sở dữ liệu kiến thức y khoa nội bộ (RAG).
-   Sử dụng các mô hình ngôn ngữ lớn (LLM) để trả lời các câu hỏi tổng quát.
-   Tra cứu thông tin từ các nguồn bên ngoài như Wikipedia và web search khi cần.
-   Lưu trữ và truy xuất lịch sử cuộc trò chuyện.

## Kiến trúc

Hệ thống được xây dựng dựa trên `LangGraph`, cho phép tạo ra một luồng xử lý có trạng thái (stateful graph) để điều phối hoạt động của các agent.

Luồng hoạt động chính:
1.  **Memory**: Nạp ngữ cảnh từ lịch sử trò chuyện.
2.  **Planner**: Phân tích câu hỏi và quyết định công cụ sẽ sử dụng (ví dụ: truy vấn cơ sở dữ liệu nội bộ hay hỏi LLM).
3.  **Tool Execution**: Các agent chuyên biệt thực thi nhiệm vụ:
    *   `RetrieverAgent`: Tìm kiếm thông tin trong VectorDB (ChromaDB) chứa kiến thức y khoa. Đây là lựa chọn ưu tiên.
    *   `LLMAgent`: Gọi đến các model LLM (Gemini, Groq...) nếu không cần truy vấn kiến thức nội bộ.
    *   `WikipediaAgent` / `TavilyAgent`: Tìm kiếm thông tin trên Internet như một phương án dự phòng.
4.  **Executor**: Tổng hợp tất cả thông tin thu thập được và tạo ra câu trả lời cuối cùng cho người dùng.

Hệ thống có cơ chế fallback linh hoạt, ví dụ nếu `RetrieverAgent` không tìm thấy thông tin, luồng sẽ tự động chuyển sang `LLMAgent` hoặc các công cụ tìm kiếm khác.

## Công nghệ sử dụng

-   **Backend**: Flask
-   **AI Framework**: LangChain, LangGraph
-   **LLM Providers**: Google Gemini, Groq, HuggingFace
-   **Vector Database**: ChromaDB cho RAG
-   **Chat History**: Supabase (PostgreSQL)
-   **Embeddings**: Sentence-Transformers
-   **Data Loaders**: PyPDF

## Cài đặt và Chạy dự án

1.  **Clone repository:**
    ```bash
    git clone https://github.com/aquaman0403/medical-chat.git
    cd medical-chat
    ```

2.  **Tạo và kích hoạt môi trường ảo:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Cấu hình môi trường:**
    -   Tạo file `.env` từ file `.env.example`.
    -   Điền các thông tin cần thiết như API keys cho Supabase, Google Gemini, Groq...

5.  **Chạy ứng dụng:**
    Lần đầu tiên chạy, ứng dụng sẽ tự động xử lý các file trong thư-mục `data` và tạo cơ sở dữ liệu vector.
    ```bash
    python app.py
    ```
    API sẽ chạy tại địa chỉ `http://127.0.0.1:8080`.

## Sử dụng API

### Health Check

-   **Endpoint**: `/`
-   **Method**: `GET`
-   **Response**:
    ```json
    {
      "status": "online",
      "service": "MEDICAL CHAT API",
      "version": "1.0.0"
    }
    ```

### Chat

-   **Endpoint**: `/api/chat`
-   **Method**: `POST`
-   **Body**:
    ```json
    {
      "conversation_id": "some-unique-session-id",
      "message": "Nguyên nhân gây ra bệnh tiểu đường là gì?"
    }
    ```
-   **Response**:
    ```json
    {
      "response": "Bệnh tiểu đường (đái tháo đường) là một bệnh rối loạn chuyển hóa mạn tính, rất phổ biến...",
      "source": "retriever",
      "timestamp": "10:30 PM",
      "success": true
    }
    ```

### Lấy lịch sử hội thoại

-   **Endpoint**: `/api/history`
-   **Method**: `GET`
-   **Query Params**: `conversation_id=<your-session-id>`
-   **Response**:
    ```json
    {
      "messages": [
        {
          "role": "user",
          "content": "Câu hỏi của người dùng...",
          "created_at": "..."
        },
        {
          "role": "assistant",
          "content": "Câu trả lời của chatbot...",
          "created_at": "..."
        }
      ]
    }
    ```
## Cấu trúc thư mục
```
.
├── app.py              # Flask App, định nghĩa API endpoints
├── main.py             # (Entry point thay thế nếu có)
├── requirements.txt    # Danh sách các thư viện
├── .env.example        # File mẫu cho biến môi trường
├── data/               # Chứa các file dữ liệu (PDF, JSON) để tạo VectorDB
├── medical_db/         # Thư mục lưu trữ ChromaDB
├── core/
│   ├── langgraph_workflow.py # "Trái tim" của dự án, định nghĩa luồng xử lý agent
│   ├── state.py            # Định nghĩa cấu trúc trạng thái của agent
│   └── database.py         # Module tương tác với Supabase (lịch sử chat)
├── agents/             # Định nghĩa logic cho từng agent chuyên biệt
└── tools/              # Các công cụ hỗ trợ (data loader, vector store, llm client)
```

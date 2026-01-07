import sys
import json
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from core.langgraph_workflow import create_workflow
from core.state import initialize_conversation_state, reset_query_state
from tools.vector_store import get_or_create_vectorstore


# Thư mục chứa file evaluation
EVALUATION_DIR = Path(__file__).parent

# File kết quả, đặt tên theo timestamp
RESULTS_FILE = EVALUATION_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Giới hạn tốc độ gọi API
DELAY_SECONDS = 12


# DANH SÁCH CÂU HỎI TEST (TIẾNG VIỆT)
TEST_QUESTIONS = [
    {
        "question": "Các triệu chứng thường gặp của bệnh cúm là gì?",
        "ground_truth_keywords": [
            "sốt", "ho", "mệt mỏi", "đau người", "đau đầu", "đau họng", "cúm"
        ]
    },
    {
        "question": "Làm thế nào để giảm huyết áp một cách tự nhiên?",
        "ground_truth_keywords": [
            "tập thể dục", "chế độ ăn", "muối", "cân nặng", "căng thẳng", "giấc ngủ", "huyết áp"
        ]
    },
    {
        "question": "Những nguyên nhân nào gây ra đau đầu?",
        "ground_truth_keywords": [
            "căng thẳng", "mất nước", "đau nửa đầu", "thiếu ngủ", "đau đầu"
        ]
    },
    {
        "question": "Sự khác nhau giữa bệnh tiểu đường type 1 và type 2 là gì?",
        "ground_truth_keywords": [
            "insulin", "tự miễn", "lối sống", "đường huyết", "tuyến tụy", "tiểu đường"
        ]
    },
    {
        "question": "Những dấu hiệu sớm của bệnh tim là gì?",
        "ground_truth_keywords": [
            "đau ngực", "khó thở", "mệt mỏi", "nhịp tim", "bệnh tim"
        ]
    },
    {
        "question": "Hen suyễn được chẩn đoán và điều trị như thế nào?",
        "ground_truth_keywords": [
            "hen suyễn", "khó thở", "thuốc xịt", "phổi", "điều trị", "chẩn đoán"
        ]
    },
    {
        "question": "Tôi bị đau ngực và khó thở. Điều này có thể là bệnh gì?",
        "ground_truth_keywords": [
            "tim", "phổi", "đau ngực", "khó thở", "lo âu", "bác sĩ", "cấp cứu"
        ]
    },
    {
        "question": "Mệt mỏi kéo dài có thể là dấu hiệu của bệnh gì?",
        "ground_truth_keywords": [
            "thiếu máu", "tuyến giáp", "trầm cảm", "giấc ngủ",
            "tiểu đường", "mãn tính", "mệt mỏi"
        ]
    },
    {
        "question": "Tại sao tôi bị chóng mặt khi đứng dậy?",
        "ground_truth_keywords": [
            "huyết áp", "tụt huyết áp", "chóng mặt", "mất nước", "thuốc", "đứng dậy"
        ]
    },
    {
        "question": "Thuốc điều trị huyết áp có những tác dụng phụ gì?",
        "ground_truth_keywords": [
            "chóng mặt", "mệt", "ho", "tiểu nhiều", "đau đầu", "tác dụng phụ", "thuốc"
        ]
    },
    {
        "question": "Kháng sinh thường mất bao lâu để có hiệu quả?",
        "ground_truth_keywords": [
            "giờ", "ngày", "kháng sinh", "đủ liều", "cải thiện", "hiệu quả"
        ]
    },
    {
        "question": "Cách điều trị trào ngược dạ dày hiệu quả nhất là gì?",
        "ground_truth_keywords": [
            "thuốc", "chế độ ăn", "tránh", "trào ngược", "dạ dày", "axit"
        ]
    },
    {
        "question": "Làm thế nào để phòng ngừa bệnh tiểu đường?",
        "ground_truth_keywords": [
            "tập thể dục", "ăn uống", "cân nặng", "đường", "tiểu đường", "phòng ngừa"
        ]
    },
    {
        "question": "Người lớn cần tiêm những loại vắc xin nào?",
        "ground_truth_keywords": [
            "cúm", "uốn ván", "viêm phổi", "viêm gan", "vắc xin", "tiêm phòng"
        ]
    },
    {
        "question": "Làm thế nào để giảm nguy cơ mắc ung thư?",
        "ground_truth_keywords": [
            "hút thuốc", "ăn uống", "tập thể dục", "rượu",
            "tầm soát", "ung thư", "phòng ngừa"
        ]
    },
]


def calculate_accuracy(response: str, ground_truth_keywords: list) -> float:
    if not response or not ground_truth_keywords:
        return 0.0

    response_lower = response.lower()
    matched = sum(1 for kw in ground_truth_keywords if kw in response_lower)

    return (matched / len(ground_truth_keywords)) * 100


def evaluate_medical_ai(app, questions: list):
    total = len(questions)

    success_count = 0
    total_time = 0
    total_words = 0
    total_accuracy = 0

    has_disclaimer = 0
    is_complete = 0
    has_source = 0

    disclaimer_keywords = [
        "tham khảo", "bác sĩ", "chuyên gia", "y tế"
    ]

    results = []
    conversation_state = initialize_conversation_state()

    print(f"\nTesting {total} questions using ground-truth evaluation\n")
    print(f"Rate limit delay: {DELAY_SECONDS}s per question")
    print(f"Estimated time: {(total * DELAY_SECONDS / 60):.1f} minutes\n")
    print("-" * 60)

    for i, q in enumerate(questions, 1):
        question = q["question"]
        ground_truth = q["ground_truth_keywords"]

        print(f"[{i}/{total}] {question}")

        conversation_state = reset_query_state(conversation_state)
        conversation_state["question"] = question

        start_time = time.time()

        try:
            result = app.invoke(conversation_state)
            conversation_state.update(result)

            response_time = time.time() - start_time
            response = result.get("generation", "")
            source = result.get("source", "unknown")

            accuracy = calculate_accuracy(response, ground_truth)
            total_accuracy += accuracy

            if response and len(response) > 20:
                success_count += 1
                print(f"     Accuracy: {accuracy:.0f}% | Source: {source}")
            else:
                print("     Empty or invalid response")

            total_time += response_time
            total_words += len(response.split()) if response else 0

            if any(word in response.lower() for word in disclaimer_keywords):
                has_disclaimer += 1

            if "tôi không biết" not in response.lower() and "không chắc" not in response.lower():
                is_complete += 1

            if source not in ["unknown", "error", "System Message"]:
                has_source += 1

            results.append({
                "question": question,
                "ground_truth_keywords": ground_truth,
                "response": response,
                "source": source,
                "accuracy": round(accuracy, 1),
                "response_time": round(response_time, 2),
                "success": bool(response and len(response) > 20)
            })

        except Exception as e:
            total_time += time.time() - start_time
            print(f"     Error: {str(e)[:100]}")

            results.append({
                "question": question,
                "ground_truth_keywords": ground_truth,
                "response": "",
                "source": "error",
                "accuracy": 0,
                "response_time": 0,
                "success": False,
                "error": str(e)
            })

        if i < total:
            print(f"     Waiting {DELAY_SECONDS}s...")
            time.sleep(DELAY_SECONDS)

    print("-" * 60)

    success_rate = (success_count / total) * 100
    avg_time = total_time / total
    avg_words = total_words / total if success_count > 0 else 0
    avg_accuracy = total_accuracy / total

    disclaimer_rate = (has_disclaimer / total) * 100
    completeness_rate = (is_complete / total) * 100
    source_rate = (has_source / total) * 100

    quality_score = (
        avg_accuracy * 0.4 +
        disclaimer_rate * 0.2 +
        completeness_rate * 0.2 +
        source_rate * 0.2
    )

    metrics = {
        "success_rate": round(success_rate, 1),
        "avg_accuracy": round(avg_accuracy, 1),
        "avg_response_time": round(avg_time, 2),
        "avg_word_count": round(avg_words, 0),
        "disclaimer_rate": round(disclaimer_rate, 1),
        "completeness_rate": round(completeness_rate, 1),
        "source_attribution": round(source_rate, 1),
        "quality_score": round(quality_score, 1)
    }

    return metrics, results


def print_results(metrics: dict):
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success Rate:        {metrics['success_rate']:.1f}%")
    print(f"Average Accuracy:    {metrics['avg_accuracy']:.1f}%")
    print(f"Average Response Time: {metrics['avg_response_time']:.2f}s")
    print(f"Average Word Count:  {metrics['avg_word_count']:.0f}")
    print(f"Disclaimer Rate:     {metrics['disclaimer_rate']:.1f}%")
    print(f"Completeness Rate:   {metrics['completeness_rate']:.1f}%")
    print(f"Source Attribution:  {metrics['source_attribution']:.1f}%")
    print("-" * 50)
    print(f"QUALITY SCORE:       {metrics['quality_score']:.1f}%")
    print("=" * 50)


def save_results(metrics: dict, results: list):
    output = {
        "evaluation_date": datetime.now().isoformat(),
        "evaluation_method": "Ground Truth Keywords Matching (Vietnamese)",
        "total_questions": len(results),
        "metrics": metrics,
        "results": results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")


def main():
    print("\n" + "=" * 60)
    print("MEDICAL CHATBOT EVALUATION")
    print("Method: Ground Truth Keywords Matching")
    print("=" * 60)

    # Khởi tạo vector database
    persist_dir = "./medical_db/"
    print("Loading vector database...")
    db = get_or_create_vectorstore(persist_dir=persist_dir)

    if db:
        print("Vector database loaded")
    else:
        print("Vector database not found")

    # Tạo workflow
    print("Creating workflow...")
    app = create_workflow()
    print("Workflow ready")

    # Chạy đánh giá
    metrics, results = evaluate_medical_ai(app, TEST_QUESTIONS)

    # In và lưu kết quả
    print_results(metrics)
    save_results(metrics, results)

    print("\nEvaluation complete")


if __name__ == "__main__":
    main()

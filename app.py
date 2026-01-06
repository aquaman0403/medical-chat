import secrets
import signal
from datetime import datetime

from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv

from core.database import SupabaseDB
from core.langgraph_workflow import create_workflow
from core.state import initialize_conversation_state, reset_query_state
from core.response import success_response, validation_error, internal_error
from tools.data_loader import process_data
from tools.vector_store import get_or_create_vectorstore

load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

workflow_app = None
db = None

MAX_MESSAGE_LENGTH = 2000
WORKFLOW_TIMEOUT = 20


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Request timeout")


signal.signal(signal.SIGALRM, timeout_handler)


def initialize_system():
    """
    Khởi tạo hệ thống: database, vectorstore và workflow.
    """
    global workflow_app, db

    pdf_path = "./data/medical_book.pdf"
    json_path = "./data/medical-data.json"
    persist_dir = "./medical_db/"

    print("Initializing Medical Chat System...")

    try:
        db = SupabaseDB()
        print("Connected to Supabase")
    except Exception as e:
        print(f"Supabase unavailable: {e}")
        db = None

    existing_db = get_or_create_vectorstore(persist_dir=persist_dir)
    if not existing_db:
        docs = process_data(pdf_path=pdf_path, json_path=json_path)
        if docs:
            get_or_create_vectorstore(documents=docs, persist_dir=persist_dir)

    workflow_app = create_workflow()
    print("System ready")


@app.route("/", methods=["GET"])
def health_check():
    return success_response(
        message="Service is running",
        data={
            "status": "online",
            "service": "MEDICAL CHAT API"
        }
    )


@app.route("/api/v1/chat", methods=["POST"])
def chat():
    global workflow_app, db

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("conversation_id") or data.get("session_id")

    if not message:
        return validation_error("No message provided")

    if len(message) > MAX_MESSAGE_LENGTH:
        return validation_error("Message too long")

    if not session_id:
        return validation_error("No conversation_id provided")

    if not workflow_app:
        return internal_error("System not initialized")

    if db:
        db.save_message(session_id, "user", message)

    conversation_state = initialize_conversation_state()

    if db:
        history = db.get_chat_history(session_id)
        conversation_state["conversation_history"] = history[-10:]

    conversation_state = reset_query_state(conversation_state)
    conversation_state["question"] = message

    try:
        signal.alarm(WORKFLOW_TIMEOUT)

        result = workflow_app.invoke(conversation_state)

        signal.alarm(0)

        response = result.get("generation", "Unable to generate response")
        source = result.get("source", "Unknown")

        if db:
            db.save_message(session_id, "assistant", response)

        return success_response(
            message="OK",
            data={
                "response": response,
                "source": source,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

    except TimeoutException:
        return internal_error("Request timed out")

    except Exception as e:
        return internal_error(str(e))


@app.route("/api/history", methods=["GET"])
def get_history():
    session_id = request.args.get("conversation_id") or request.args.get("session_id")

    if not session_id:
        return validation_error("No conversation_id provided")

    if not db:
        return success_response("No database", {"messages": []})

    messages = db.get_chat_history(session_id)
    return success_response("OK", {"messages": messages})


if __name__ == "__main__":
    initialize_system()
    app.run(debug=False, port=8080)

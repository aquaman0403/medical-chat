import secrets
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from core.database import SupabaseDB
from core.langgraph_workflow import create_workflow
from core.state import initialize_conversation_state
from core.state import reset_query_state
from core.response import (
    success_response, validation_error, internal_error, bad_request
)
from tools.data_loader import process_data
from tools.vector_store import get_or_create_vectorstore

load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)  # Enable CORS for all routes

# Global workflow and conversation states
workflow_app = None
conversation_states = {}
db = None


def initialize_system():
    global workflow_app, db

    pdf_path = './data/medical_book.pdf'
    json_path = './data/medical-data.json'
    persist_dir = './medical_db/'

    print("Initializing Medical Chat System...")

    # Initialize Supabase Database
    try:
        db = SupabaseDB()
        print("Connected to Supabase...")
    except Exception as e:
        print(f"Failed to connect to Supabase: {e}")
        print("Chat history will not be saved!")

    # Try to load an existing database
    existing_db = get_or_create_vectorstore(persist_dir=persist_dir)

    if not existing_db:
        print("Creating vector database from data sources...")
        doc_splits = process_data(pdf_path=pdf_path, json_path=json_path)
        if doc_splits:
            get_or_create_vectorstore(documents=doc_splits, persist_dir=persist_dir)
        else:
            print("No documents found to create database")

    workflow_app = create_workflow()
    print("Medical Chat API Ready!")


@app.route('/', methods=['GET'])
def health_check():
    return success_response(
        message="Service is running",
        data={
            "status": "online",
            "service": "MEDICAL CHAT API",
            "version": "1.0.0"
        }
    )


@app.route('/api/v1/chat', methods=['POST'])
def chat():
    global workflow_app, conversation_states, db

    data = request.json
    message = data.get('message', '')
    session_id = data.get('conversation_id') or data.get('session_id')

    if not message:
        return validation_error(message='No message provided')

    if not session_id:
        return validation_error(message='No conversation_id provided')

    if not workflow_app:
        return internal_error(message='System not initialized')

    # Save user message to database
    if db:
        db.save_message(session_id, 'user', message)

    # Initialize or get conversation state
    if session_id not in conversation_states:
        conversation_states[session_id] = initialize_conversation_state()
    
    # Load last 10 messages (5 user + 5 assistant) from DB for context
    if db:
        history = db.get_chat_history(session_id)
        # Get last 10 messages for context (5 Q&A pairs)
        recent_history = history[-10:] if len(history) >= 10 else history
        conversation_states[session_id]["conversation_history"] = recent_history

    conversation_state = conversation_states[session_id]
    conversation_state = reset_query_state(conversation_state)
    conversation_state["question"] = message

    # Process query through workflow
    try:
        result = workflow_app.invoke(conversation_state)
        conversation_states[session_id].update(result)

        # Get current UTC timestamp in ISO 8601 format
        timestamp = datetime.utcnow().isoformat() + 'Z'

        # Extract response and source
        response = result.get('generation', 'Unable to generate response.')
        source = result.get('source', 'Unknown')

        # Save assistant response to database
        if db:
            db.save_message(session_id, 'assistant', response)

        return success_response(
            message="Chat response generated successfully",
            data={
                'response': response,
                'source': source,
                'timestamp': timestamp
            }
        )
    except Exception as e:
        print(f"Error processing chat: {e}")
        return internal_error(message=str(e))


@app.route('/api/history', methods=['GET'])
def get_history():
    global db
    session_id = request.args.get('conversation_id') or request.args.get('session_id')

    if not session_id:
        return validation_error(message='No conversation_id provided')

    if db:
        messages = db.get_chat_history(session_id)
        return success_response(
            message="Chat history retrieved successfully",
            data={'messages': messages}
        )

    return success_response(
        message="No database connection",
        data={'messages': []}
    )


if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, port=8080)

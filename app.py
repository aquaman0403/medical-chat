import secrets
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from core.database import SupabaseDB
from core.langgraph_workflow import create_workflow
from core.state import initialize_conversation_state
from core.state import reset_query_state
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
    return jsonify({
        "status": "online",
        "service": "MEDICAL CHAT API",
        "version": "1.0.0"
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    global workflow_app, conversation_states, db

    data = request.json
    message = data.get('message', '')
    session_id = data.get('conversation_id') or data.get('session_id')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    if not session_id:
        return jsonify({'error': 'No conversation_id provided'}), 400

    if not workflow_app:
        return jsonify({'error': 'System not initialized'}), 500

    # Save user message to database
    if db:
        db.save_message(session_id, 'user', message)

    # Initialize or get conversation state
    if session_id not in conversation_states:
        # Try to load history from DB for context
        history = []
        if db:
            history = db.get_chat_history(session_id)

        conversation_states[session_id] = initialize_conversation_state()
        # You might want to populate state["messages"] with history here if your agent supports it

    conversation_state = conversation_states[session_id]
    conversation_state = reset_query_state(conversation_state)
    conversation_state["question"] = message

    # Process query through workflow
    try:
        result = workflow_app.invoke(conversation_state)
        conversation_states[session_id].update(result)

        # Get current timestamp
        timestamp = datetime.now().strftime("%I:%M %p")

        # Extract response and source
        response = result.get('generation', 'Unable to generate response.')
        source = result.get('source', 'Unknown')

        # Save assistant response to database
        if db:
            db.save_message(session_id, 'assistant', response)

        return jsonify({
            'response': response,
            'source': source,  # Returning source to frontend even if not saved to DB
            'timestamp': timestamp,
            'success': bool(result.get('generation'))
        })
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    global db
    session_id = request.args.get('conversation_id') or request.args.get('session_id')

    if not session_id:
        return jsonify({'error': 'No conversation_id provided'}), 400

    if db:
        messages = db.get_chat_history(session_id)
        return jsonify({'messages': messages})

    return jsonify({'messages': []})


if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, port=8080)

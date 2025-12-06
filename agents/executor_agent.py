from core.state import AgentState
from core.prompts import get_rag_prompt
from tools.llm_client import LLMClient

FALLBACK_RESPONSE = "Tôi hiểu lo lắng của bạn về triệu chứng này. Để được tư vấn y tế chính xác, vui lòng tham khảo ý kiến chuyên gia y tế có thể đánh giá đúng tình trạng của bạn. Các thông tin mà chatbot cung cấp chỉ mang tính chất tham khảo. Hãy thật cẩn thận với các thông tin này."

def _add_to_history(state: AgentState, question: str, answer: str, source: str):
    """Helper function to add Q&A to conversation history"""
    state["conversation_history"].append({
        'role': 'user',
        'content': question
    })
    state["conversation_history"].append({
        'role': 'assistant',
        'content': answer,
        'source': source
    })

def ExecutorAgent(state: AgentState) -> AgentState:
    question = state["question"]
    source_info = state.get("source", "Unknown")

    # Get conversation context (5 cặp Q&A gần nhất)
    history_context = ""
    for item in state.get("conversation_history", [])[-10:]:
        if item.get('role') == 'user':
            history_context += f"Người dùng: {item.get('content', '')}\n"
        elif item.get('role') == 'assistant':
            history_context += f"MedicalBot: {item.get('content', '')}\n"

    # If LLM was successful earlier (from LLMAgent), use that response
    if state.get("llm_success", False) and state.get("generation"):
        answer = state["generation"]
        _add_to_history(state, question, answer, source_info)
        print("Executor: Using LLM response from earlier")
        return state

    # If we have documents from retrieval, generate response with RAG
    if state.get("documents") and len(state["documents"]) > 0:
        try:
            llm = LLMClient.get_llm()
            if not llm:
                raise Exception("LLM client not available")
                
            content = "\n\n".join([doc.page_content[:1000] for doc in state["documents"][:3]])
            prompt = get_rag_prompt(history_context, question, content)

            response = llm.invoke(prompt)
            answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

            if answer and len(answer) > 10:
                state["generation"] = answer
                state["source"] = source_info
                _add_to_history(state, question, answer, source_info)
                print("Executor: Generated response with RAG documents")
                return state
            else:
                print("Executor: RAG response too short, using fallback")
        except Exception as e:
            print(f"Executor: Error calling Gemini API - {e}")

    # Fallback response when all else fails
    print("Executor: Using fallback response")
    state["generation"] = FALLBACK_RESPONSE
    state["source"] = "System Message"
    _add_to_history(state, question, FALLBACK_RESPONSE, "System Message")

    return state

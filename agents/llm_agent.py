from core.state import AgentState
from core.prompts import get_llm_prompt
from tools.llm_client import LLMClient

def LLMAgent(state: AgentState) -> AgentState:
    try:
        llm = LLMClient.get_llm()
        
        if not llm:
            print("LLM: No LLM client available")
            state["llm_success"] = False
            state["llm_attempted"] = True
            return state
        
        history_context = ""
        for item in state.get("conversation_history", [])[-5:]:
            if item.get('role') == 'user':
                history_context += f"Người dùng: {item.get('content', '')}\n"
            elif item.get('role') == 'assistant':
                history_context += f"MedicalBot: {item.get('content', '')}\n"
        
        prompt = get_llm_prompt(history_context, state['question'])

        response = llm.invoke(prompt)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        if answer and len(answer) > 10:
            state["generation"] = answer
            state["llm_success"] = True
            state["source"] = "AI Medical Knowledge"
            print("LLM: Successfully generated response")
        else:
            state["llm_success"] = False
            print("LLM: Response too short or empty")

    except Exception as e:
        print(f"LLM: Error calling Gemini API - {e}")
        state["llm_success"] = False

    state["llm_attempted"] = True
    return state

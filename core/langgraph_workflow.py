from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.memory_agent import MemoryAgent
from agents.planner_agent import PlannerAgent
from agents.llm_agent import LLMAgent
from agents.retriever_agent import RetrieverAgent
from agents.wikipedia_agent import WikipediaAgent
from agents.tavily_agent import TavilyAgent
from agents.executor_agent import ExecutorAgent


def route_after_planner(state: AgentState):
    if state["current_tool"] == "retriever":
        return "retriever"
    return "llm_agent"


def route_after_llm(state: AgentState):
    if state.get("llm_success", False):
        return "executor"
    # Chỉ chuyển sang retriever nếu chưa thử RAG
    if not state.get("rag_attempted", False):
        return "retriever"
    # Nếu LLM và RAG đều thất bại, thử Wikipedia
    return "wikipedia"


def route_after_rag(state: AgentState):
    if state.get("rag_success", False):
        return "executor"
    # Chỉ thử LLM nếu chưa thử trước đó
    if not state.get("llm_attempted", False):
        return "llm_agent"
    # Nếu RAG và LLM đều thất bại, thử Wikipedia
    return "wikipedia"


def route_after_wiki(state: AgentState):
    if state.get("wiki_success", False):
        return "executor"
    return "tavily"


def route_after_tavily(state: AgentState):
    return "executor"


def create_workflow():
    workflow = StateGraph(AgentState)

    # Khai báo các node trong workflow
    workflow.add_node("memory", MemoryAgent)
    workflow.add_node("planner", PlannerAgent)
    workflow.add_node("llm_agent", LLMAgent)
    workflow.add_node("retriever", RetrieverAgent)
    workflow.add_node("wikipedia", WikipediaAgent)
    workflow.add_node("tavily", TavilyAgent)
    workflow.add_node("executor", ExecutorAgent)

    workflow.set_entry_point("memory")

    workflow.add_edge("memory", "planner")

    # Điều hướng sau planner
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "llm_agent": "llm_agent"
        }
    )

    # Điều hướng sau LLM
    workflow.add_conditional_edges(
        "llm_agent",
        route_after_llm,
        {
            "executor": "executor",
            "retriever": "retriever",
            "wikipedia": "wikipedia"
        }
    )

    # Điều hướng sau retriever (RAG)
    workflow.add_conditional_edges(
        "retriever",
        route_after_rag,
        {
            "executor": "executor",
            "llm_agent": "llm_agent",
            "wikipedia": "wikipedia"
        }
    )

    # Điều hướng sau Wikipedia
    workflow.add_conditional_edges(
        "wikipedia",
        route_after_wiki,
        {
            "executor": "executor",
            "tavily": "tavily"
        }
    )

    # Điều hướng sau Tavily
    workflow.add_conditional_edges(
        "tavily",
        route_after_tavily,
        {
            "executor": "executor"
        }
    )

    workflow.add_edge("executor", END)

    return workflow.compile()

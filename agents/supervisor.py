import os

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.state import ResearchState
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node
from memory.store import (
    get_user_preferences,
    get_query_history,
    append_query,
)

class _Plan(BaseModel):
    """Structured output schema for the planner."""
    subtasks: list[str] = Field(
        description=(
            "An ordered JSON array of sub-task strings (1–4 entries). "
            "Each element MUST be a string. Do NOT return a single concatenated string."
        ),
    )


_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You decompose research questions into 1–4 ordered, independently-"
     "answerable sub-tasks. Prefer fewer, larger sub-tasks over many tiny "
     "ones. Each sub-task should be answerable from a single retrieval.\n\n"
     "Output schema: return JSON with a single key 'subtasks' whose value is "
     "a JSON array of strings. Never return a single concatenated string."),
    ("human",
     "User preferences: {preferences}\n"
     "Recent past questions from this user: {history}\n\n"
     "New question: {question}\n\n"
     "Return the sub-tasks as a JSON list of strings."),
])


def planner_node(state: ResearchState) -> dict:
    user_id = state.get("user_id", "default")
    prefs = get_user_preferences(user_id)
    history = get_query_history(user_id, limit=3)
    append_query(user_id, state["question"])

    llm = ChatBedrock(
        model_id=os.environ["BEDROCK_MODEL_ID"],
        provider="anthropic",
        region_name=os.environ["AWS_REGION"],
        model_kwargs={"max_tokens": 512, "temperature": 0.0},
    )
    chain = _PLANNER_PROMPT | llm.with_structured_output(_Plan)
    plan = chain.invoke({
        "question": state["question"],
        "preferences": prefs,
        "history": history or ["<none>"],
    })

    return {
        "plan": plan.subtasks,
        "current_subtask_index": 0,
        "iteration_count": 0,
        "needs_hitl": False,
        "scratchpad": [f"[planner] decomposed into {len(plan.subtasks)} sub-tasks"],
    }


def router(state: ResearchState) -> str:
    """Conditional-edge function. MUST return the name of the next node."""
    if not state.get("retrieved_chunks"):
        return "retriever"
    if not state.get("analysis"):
        return "analyst"
    if not state.get("fact_check_report"):
        return "fact_checker"
    return "critique"


def critique_node(state: ResearchState) -> dict:
    """Decide: accept, retry, or escalate to HITL."""
    iteration = state.get("iteration_count", 0) + 1
    confidence = state.get("confidence_score", 0.0)
    threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    max_iter = int(os.environ.get("MAX_REFINEMENT_ITERATIONS", 3))

    log = [f"[critique] iter={iteration}, conf={confidence:.2f}, "
           f"threshold={threshold}, max_iter={max_iter}"]

    # Path 1: confident enough → accept.
    if confidence >= threshold and not state.get("needs_hitl"):
        log.append("[critique] accepted")
        return {"iteration_count": iteration, "scratchpad": log}

    # Path 2: budget exhausted → escalate.
    if iteration >= max_iter:
        log.append("[critique] max iterations reached — escalating to HITL")
        # NodeInterrupt pauses the graph; resume by graph.update_state(...).
        raise NodeInterrupt(
            f"Confidence {confidence:.2f} below threshold {threshold} "
            f"after {iteration} iterations. Human review required."
        )

    # Path 3: retry — clear downstream state so the router re-runs them.
    log.append("[critique] retrying — clearing analysis & fact_check")
    return {
        "iteration_count": iteration,
        "retrieved_chunks": [],          # forces retriever to re-fetch
        "analysis": {},                  # forces analyst to re-synthesize
        "fact_check_report": {},
        "scratchpad": log,
    }


def _critique_router(state: ResearchState) -> str:
    """Edge after critique_node — END if accepted, else loop."""
    confidence = state.get("confidence_score", 0.0)
    threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    if confidence >= threshold and not state.get("needs_hitl"):
        return END
    return "retriever"


def build_supervisor_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("critique", critique_node)

    graph.add_edge(START, "planner")
    # After planner, the router picks whichever specialist is needed first
    # (almost always the retriever). Listed below so LangGraph knows the
    # full set of valid destinations.
    graph.add_conditional_edges(
        "planner", router,
        {"retriever": "retriever", "analyst": "analyst",
         "fact_checker": "fact_checker", "critique": "critique"},
    )
    # Linear within a sub-task: retriever → analyst → fact_checker → critique
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "fact_checker")
    graph.add_edge("fact_checker", "critique")

    # Critique decides whether to loop or end.
    graph.add_conditional_edges(
        "critique", _critique_router,
        {"retriever": "retriever", END: END},
    )

    # MemorySaver = in-process checkpointer; switch to a persistent saver
    # (PostgresSaver / DynamoDBSaver) when deploying to Lambda.
    return graph.compile(checkpointer=MemorySaver())
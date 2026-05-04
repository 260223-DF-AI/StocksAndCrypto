"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""
import os

from agents.state import ResearchState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
#from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node

from openai import OpenAI

from memory.store import (
    get_user_preferences,
    get_query_history,
    append_query
)
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

class _Plan(BaseModel):
    """Structured output schema for the planner."""
    subtasks: list[str] = Field(
        description=(
            "An ordered JSON array of sub-task strings (1-4 entries)."
            "Each element MUST be a string. Do NOT return a single concatenated string."
        )
    )

_Planner_Prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You decompose research questions into 1-4 ordered, independently-"
     "answerable sub-tasks. Prefer fewer, larger sub-tasks over many tiny "
     "ones. Each sub-task should be answerable from a single retrieval.\n\n"
     "Output schema: return JSON with a single key 'subtasks' whose value is "
     "a JSON array of strings. Never return a single concatenated string."),
    ("human",
     "User preferences: {preferences}\n"
     "Recent past questions from this user: {history}\n\n"
     "New question: {question}\n\n"
     "Return the sub-tasks as a JSON list of strings.")
])

def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    user_id = state.get("user_id", "default")
    prefs = get_user_preferences(user_id)
    history = get_query_history(user_id, limit = 3)
    append_query(user_id, state["question"])

    # chat_model = ChatBedrock(
    #     model_id = os.environ["BEDROCK_MODEL_ID"],
    #     region_name = os.environ["AWS_REGION"],
    #     model_kwargs = {
    #         "max_tokens": 512,
    #         "temperature": 0.0
    #     }
    # )
    
    # setup = _Planner_Prompt | chat_model.with_structured_output(_Plan)
    # plan = setup.invoke({
    #     "question": state["question"],
    #     "preferences": prefs,
    #     "history": history or ["<none>"]
    # })
    
    """
            You decompose research questions into 1-4 ordered, independently-
            answerable sub-tasks. Prefer fewer, larger sub-tasks over many tiny
            ones. Each sub-task should be answerable from a single retrieval.\n\n
            Output schema: return JSON with a single key 'subtasks' whose value is
            a JSON array of strings. Never return a single concatenated string."
            
            User preferences: {prefs}\n"
            Recent past questions from this user: {history}\n\n"
            New question: {state['question']}\n\n"

            Return an ordered JSON array of sub-task strings (1-4 entries).
            Each element MUST be a string. Do NOT return a single concatenated string.
            Return the sub-tasks as a JSON list of strings."
        """
    
    plan = client.responses.create(
        model="gpt-5-nano",
        input=        
        """
            You decompose research questions into ordered independently executable sub-tasks.
            Based on the user's question, return a list of sub-tasks (Plan-and-Execute pattern).
            
            These are the sub-tasks that you have access to:
            retriever: get relevant embeddings from a Pinecone index\n
            analyst: provide a detailed response based on retrieved chunks and question\n
            fact_checker: validate the analyst's response is factually correct\n

            Output: a list of sub-tasks as a JSON array of strings.
            
            Example Output:
            Question: "How often will a single person mine a block of Bitcoin?"

            Output:
            "{
                {retriever: the process of mining a block}, 
                {retriever: bitcoin},
                {analyst},
                {fact_checker},
                {critique}
            }"
            
            Question:
        """
        + state["question"]
        )
    print(plan)
    print(plan.output[1].content[0].text)
    return {
        "plan": plan.output[1].content[0].text,
        "current_sub_task_idx": 0,
        "iteration_count": 0,
        "hitl": False,
        "scratchpad": [f"[planner] decomposed into {len(plan.output[1].content[0].text)} sub-tasks"]
    }


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    if not state.get("retrieved_chunks"):
        return "retriever"
    if not state.get("analysis"):
        return "analyst"
    if not state.get("fact_check_report"):
        return "fact_checker"
    return "critique"


def critique_node(state: ResearchState) -> dict:
    """
    Evaluate the aggregated response and decide: accept, retry, or escalate.

    TODO:
    - Check confidence_score against the HITL threshold.
    - If below threshold and iterations < max, loop back for refinement.
    - If below threshold and iterations >= max, trigger HITL interrupt.
    - If above threshold, accept and route to END.
    - Increment iteration_count.
    """
    iteration = state.get("iteration_count", 0) + 1
    confidence_threshold = state.get("confidence_score", 0.0)
    threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    max_iterations = int(os.environ.get("MAX_REFINEMENT_ITERATIONS", 3))

    # Path 1: confident enough, can accept
    log = [f"[critique] iter={iteration}, confidence={confidence_threshold:.2f},"
           f"threshold={threshold}, max_iter={max_iterations}"]
    if confidence_threshold >= threshold and not state.get("hitl"):
        log.append("[critique] accepted")
        return {"iteration_count": iteration, "scratchpad": log}
    
    #Path 2: budget exhausted, escalate
    if iteration >= max_iterations:
        log.append("[critique] max iterations reached, HITL triggered")
        raise NodeInterrupt(
            f"Confidence {confidence:.2f} below threshold {threshold}"
            f"after {iteration} iterations. Human review required."
        )
    
    # Path 3: retry, clear downstream state so the router re-runs
    log.append("[critique] retrying - clearing analysis & fact check")
    return {
        "iteration_count": iteration,
        "retrieved_chunks": [],
        "analysis": {},
        "fact_check_report": {},
        "scratchpad": log
    }

def _critique_router(state: ResearchState) -> str:
    """Edge after critique_node, END if accepted, otherwise loop."""
    confidence = state.get("confidence_score", 0.0)
    threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    if confidence >= threshold and not state.get("hitl"):
        return END
    return "retriever"

def build_supervisor_graph():
    """
    Construct and compile the Supervisor StateGraph.

    TODO:
    - Instantiate StateGraph with ResearchState.
    - Add nodes: planner, retriever, analyst, fact_checker, critique.
    - Add edges and conditional edges (router).
    - Set entry point to planner.
    - Compile and return the graph.

    Returns:
        A compiled LangGraph that can be invoked with an initial state.
    """
    workflow = StateGraph(ResearchState)    # instantiate stategraph with our ResearchState schema

    workflow.add_node("planner", planner_node) # add planner node to the graph
    workflow.add_node("retriever", retriever_node) # add retriever node to the graph
    workflow.add_node("analyst", analyst_node) # add analyst node to the graph
    workflow.add_node("fact_checker", fact_checker_node) # add fact checker node to the graph
    workflow.add_node("critique", critique_node) # add critique node to the graph

    workflow.add_edge(START, "planner") # set entry point to planner

    # Router will pick which specialist is needed
    workflow.add_conditional_edges(
        "planner", router,
        {"retriever": "retriever", "analyst": "analyst",
        "fact_checker": "fact_checker", "critique": "critique"}
    )
    workflow.add_edge("retriever", "analyst")
    workflow.add_edge("analyst", "fact_checker")
    workflow.add_edge("fact_checker", "critique")

    # Critique will be the decider for ending or looping
    workflow.add_conditional_edges(
        "critique", critique_node,
        {"retriever": "retriever", END: END}
    )

    return workflow.compile(checkpointer=MemorySaver())
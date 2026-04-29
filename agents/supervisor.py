"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from agents.state import ResearchState
from langgraph.graph import StateGraph
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node
from langchang.bedrock import ChatBedrock

def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    chat_model = ChatBedrock(
        model_id = "anthropic.claude-3-haiku-20240307-v1:0",
        region_name = "us-east-1",
        model_kwargs = {
            "temperature": 0.1
        }
    )
    prompt = f"""You are the Planner agent in a research workflow. 
    Your task is to analyze the question and decompose it into actionable sub-tasks.
    Return only a list of sub-tasks (Plan-and-Execute pattern).
    These are the actions that can be in the list of sub-tasks:
    - retrieve: gather more information or data
    - analyze: perform analysis or synthesis on the retrieved information
    - fact_check: verify claims against trusted sources
    - critique: evaluate and refine the response
    Question: {state["question"]}
    """
    plan = chat_model(prompt)
    state["plan"] = plan
    state["scratchpad"].append(f"Plan: {plan}")
    return {"plan": plan}


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    current_plan = state["plan"]
    if "retrieve" in current_plan:
        return "retriever"
    elif "analyze" in current_plan:
        return "analyst"
    elif "fact_check" in current_plan:
        return "fact_checker"
    elif "critique" in current_plan:
        return "critique"
    else:
        raise ValueError("Router could not determine the next node based on the plan.")


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
    confidence_threshold = 0.7
    max_iterations = 3
    if state["confidence_score"] < confidence_threshold:
        if state["iteration_count"] < max_iterations:
            state["iteration_count"] += 1
            return "planner"  # route back to planner for refinement
        else:
            raise Exception("HITL Interrupt: Confidence below threshold after max iterations.")
    else:
        state["iteration_count"] += 1
        return "end"


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

    workflow.add_node(planner_node, name="planner") # add planner node to the graph
    workflow.add_node(retriever_node, name="retriever") # add retriever node to the graph
    workflow.add_node(analyst_node, name="analyst") # add analyst node to the graph
    workflow.add_node(fact_checker_node, name="fact_checker") # add fact checker node to the graph
    workflow.add_node(critique_node, name="critique") # add critique node to the graph

    workflow.add_conditional_edge("planner", "router", condition=lambda state: True) # always route to router
    workflow.add_conditional_edge("router", "retriever", condition=lambda state: "retrieve" in state["plan"]) # if router decides we need to retrieve more info, route to retriever
    workflow.add_conditional_edge("router", "analyst", condition=lambda state: "analyze" in state["plan"]) # if router decides we need to analyze, route to analyst
    workflow.add_conditional_edge("router", "fact_checker", condition=lambda state: "fact_check" in state["plan"]) # if router decides we need to fact check, route to fact checker
    workflow.add_conditional_edge("router", "critique", condition=lambda state: "critique" in state["plan"]) # if router decides we need to critique, route to critique

    workflow.add_conditional_edge("critique", "planner", condition=lambda state: state["iteration_count"] < 3) # if critique decides to retry, route back to planner

    workflow.set_entry_point("planner") # entry point to the graph

    return workflow.compile()
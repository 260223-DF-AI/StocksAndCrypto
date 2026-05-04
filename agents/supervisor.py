"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from agents.state import ResearchState
from langgraph.graph import StateGraph, START
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node
from dotenv import load_dotenv
#import os
from openai import OpenAI
load_dotenv()
client = OpenAI()

def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """

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
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt)
    #plan = chat_model.invoke(prompt)
    plan = response
    state["plan"] = plan
    print(plan)
    #state["scratchpad"].append(f"Plan: {plan}")
    #print(plan)
    return {"plan": plan}


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    current_plan = state["plan"][-1]
    if "retrieve" == current_plan:
        return "retriever"
    elif "analyze" == current_plan:
        return "analyst"
    elif "fact_check" == current_plan:
        return "fact_checker"
    elif "critique" == current_plan:
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

    workflow.add_node("planner", planner_node) # add planner node to the graph
    workflow.add_node("retriever", retriever_node) # add retriever node to the graph
    workflow.add_node("analyst", analyst_node) # add analyst node to the graph
    workflow.add_node("fact_checker", fact_checker_node) # add fact checker node to the graph
    workflow.add_node("critique", critique_node) # add critique node to the graph

    workflow.add_edge(START, "planner") # set entry point to planner

    workflow.add_conditional_edges("planner", router) # add conditional edge from planner to router
    workflow.add_conditional_edges("retriever", router) # add conditional edge from retriever to router
    workflow.add_conditional_edges("analyst", router) # add conditional edge from analyst to router
    workflow.add_conditional_edges("fact_checker", router) # add conditional edge from fact checker to router
    workflow.add_conditional_edges("critique", router) # add conditional edge from critique to router

    return workflow.compile()
"""
ResearchFlow — Graph State Definition

Defines the TypedDict that flows through the Supervisor StateGraph.
All nodes read from and write to this shared state.
"""

from typing import TypedDict, Annotated
from operator import add


class ResearchState(TypedDict):
    """
    Shared state for the Supervisor graph.

    TODO: Expand these fields as your design evolves.

    Attributes:
        question: The original user research question.
        plan: Decomposed sub-tasks from the Planner node.
        retrieved_chunks: Chunks returned by the Retriever agent.
        analysis: Synthesized response from the Analyst agent.
        fact_check_report: Verification report from the Fact-Checker agent.
        confidence_score: Overall confidence in the final answer (0.0–1.0).
        iteration_count: Number of self-refinement loops executed so far.
        scratchpad: Step-wise log of intermediate outputs for observability.
        user_id: Identifier for cross-thread memory via the Store interface.
    """
    question: str
    user_id: str
    plan: list[str]
    current_sub_plan_idx: int
    retrieved_chunks: list[dict]
    analysis: dict
    fact_check_report: dict
    confidence_score: float
    iteration_count: int
    hitl: bool  # true -> critique routes to HITL interrupt
    scratchpad: Annotated[list[str], add]

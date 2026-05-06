import json
import logging
import os
import uuid

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Lazy import + module-level so Lambda re-uses warm-state across invocations.
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        # Import inside the function so cold-start cost is only paid once
        # per container, not at every cold init.
        from agents.supervisor import build_supervisor_graph
        _graph = build_supervisor_graph()
    return _graph


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event: dict, context) -> dict:
    """POST /research handler."""
    try:
        # API Gateway proxy integration delivers the body as a string.
        raw_body = event.get("body") or "{}"
        if isinstance(raw_body, str):
            body = json.loads(raw_body)
        else:
            body = raw_body

        question = (body.get("question") or "").strip()
        if not question:
            return _response(400, {"error": "Missing required field 'question'"})

        # --- input boundary ---
        from middleware.guardrails import detect_injection, sanitize_input
        from middleware.pii_masking import mask_pii
        if detect_injection(question):
            return _response(400, {"error": "Input rejected: possible prompt injection"})
        question = mask_pii(sanitize_input(question))

        # --- invoke graph ---
        graph = _get_graph()
        config = {"configurable": {"thread_id": f"lambda-{uuid.uuid4()}"}}
        result = graph.invoke(
            {"question": question, "user_id": body.get("user_id", "anonymous")},
            config=config,
        )

        # --- output boundary + response shape ---
        analysis = result.get("analysis", {}) or {}
        return _response(200, {
            "answer": mask_pii(analysis.get("answer", "")),
            "citations": analysis.get("citations", []),
            "confidence": result.get("confidence_score", 0.0),
            "fact_check": result.get("fact_check_report", {}),
            "iterations": result.get("iteration_count", 0),
        })

    except Exception as e:
        logger.exception("research failed")
        return _response(500, {"error": str(e)})
"""
Unit Tests — Analyst Agent

Tests the analyst node using mocked Bedrock calls.
Validates structured output schema and confidence scoring.
"""

from unittest.mock import patch, MagicMock

import pytest

from agents.analyst import AnalysisResult, Citation


class TestAnalystAgent:
    """Tests for agents.analyst.analyst_node."""

    def _stub_result(self):
        return AnalysisResult(
            answer="The stock market is a broad term for the network of exchanges and over-the-counter venues where investors buy and sell shares in publicly traded companies through brokerage platforms [1].",
            citations=[Citation(source = "What is the Stock Market.txt", page_number=1,
                                excerpt = "The stock market is a broad term for the network of exchanges and over-the-counter venues where investors buy and sell shares in publicly traded companies through brokerage platforms..")],
            confidence = 0.88
        )

    def test_returns_valid_analysis_result(self):
        """
        TODO:
        - Mock the Bedrock LLM invocation.
        - Call analyst_node with sample retrieved_chunks.
        - Assert the output parses into a valid AnalysisResult.
        """
        with patch("agents.analyst.ChatBedrock") as MockChat, \
             patch("agents.analyst._PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = self._stub_result()

            instance = MockChat.return_value
            instance.with_structured_output.return_value = MagicMock()

            mock_prompt.__or__.return_value = mock_chain

            from agents.analyst import analyst_node
            out = analyst_node({
                "question": "What is the stock market?",
                "plan": ["What is the stock market?"],
                "current_subtask_index": 0,
                "retrieved_chunks": [
                    {"content": "The stock market is a broad term for the network of exchanges and over-the-counter venues where investors buy and sell shares in publicly traded companies through brokerage platforms.",
                     "source": "What is the Stock Market.txt", "page_number": 1,
                     "relevance_score": 0.91}
                ]
            })
            assert "analysis" in out
            assert out["confidence_score"] == 0.88
            assert out["analysis"]["citations"][0]["source"] == "What is the Stock Market.txt"


    def test_includes_citations(self):
        """
        TODO:
        - Assert the AnalysisResult contains at least one Citation.
        - Assert citation source matches a retrieved chunk source.
        """
        with patch("agents.analyst.ChatBedrock") as MockChat, \
             patch("agents.analyst._PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = self._stub_result()

            instance = MockChat.return_value
            instance.with_structured_output.return_value = MagicMock()

            mock_prompt.__or__.return_value = mock_chain
            from agents.analyst import analyst_node
            out = analyst_node({
                "question": "What is the stock market?",
                "plan": ["What is the stock market?"],
                "current_subtask_index": 0,
                "retrieved_chunks": [
                    {"content": "The stock market is a broad term for the network of exchanges and over-the-counter venues where investors buy and sell shares in publicly traded companies through brokerage platforms.",
                     "source": "What is the Stock Market.txt", "page_number": 1,
                     "relevance_score": 0.91}
                ]
            })
            assert len(out["analysis"]["citations"]) > 0
            assert out["analysis"]["citations"][0]["source"] == "What is the Stock Market.txt"

    def test_confidence_within_range(self):
        """
        TODO:
        - Assert confidence_score is between 0.0 and 1.0.
        """
        with patch("agents.analyst.ChatBedrock") as MockChat, \
             patch("agents.analyst._PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = self._stub_result()

            instance = MockChat.return_value
            instance.with_structured_output.return_value = MagicMock()

            mock_prompt.__or__.return_value = mock_chain
            from agents.analyst import analyst_node
            out = analyst_node({
                "question": "What is the stock market?",
                "plan": ["What is the stock market?"],
                "current_subtask_index": 0,
                "retrieved_chunks": [
                    {"content": "The stock market is a broad term for the network of exchanges and over-the-counter venues where investors buy and sell shares in publicly traded companies through brokerage platforms.",
                     "source": "What is the Stock Market.txt", "page_number": 1,
                     "relevance_score": 0.91}
                ]
            })
            assert 0.0 <= out["confidence_score"] <= 1.0

    def test_short_circuits_when_no_chunks(self):
        from agents.analyst import analyst_node
        out = analyst_node({
            "question": "x",
            "plan": ["x"],
            "current_subtask_index": 0,
            "retrieved_chunks": [],
        })
        assert out["confidence_score"] == 0.0
        assert out["analysis"]["citations"] == []
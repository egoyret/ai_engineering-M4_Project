"""
test_extraction_agent.py
-------------------------
Verifica que run_extraction_agent() lanza ExtractionError
en todos los escenarios de fallo esperados.

Estrategia de mocking:
  - 'src.agents.extraction_agent.ChatOpenAI': patcheamos la clase en el módulo destino.
  - El LLM usa .with_structured_output(include_raw=True), por lo que el mock necesita
    devolver un Runnable cuyo invoke retorne el dict {parsed, raw, parsing_error}.
  - Usamos RunnableLambda para los casos de excepción y MagicMock con
    with_structured_output configurado para los casos de dict de error.
  - 'langfuse': fixture de conftest.py.

No se realizan llamadas reales a la API de OpenAI.
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException

from src.agents.extraction_agent import run_extraction_agent
from src.exceptions import ExtractionError


def _make_mock_llm_with_structured_output(invoke_behavior):
    """
    Helper que construye un mock de ChatOpenAI donde
    llm.with_structured_output(...) devuelve un RunnableLambda
    con el comportamiento indicado.

    Args:
        invoke_behavior: función (input) → result o raise
    """
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(invoke_behavior)
    return mock_llm


class TestExtractionOutputParserException:
    """OutputParserException durante la invocación de la chain → ExtractionError."""

    def test_output_parser_exception_raises_extraction_error(
        self, sample_texts, mock_langfuse
    ):
        def _raise_output_parser(_):
            raise OutputParserException("El LLM no produjo JSON válido.")

        mock_llm = _make_mock_llm_with_structured_output(_raise_output_parser)

        with patch("src.agents.extraction_agent.ChatOpenAI", return_value=mock_llm):
            with pytest.raises(ExtractionError) as exc_info:
                run_extraction_agent(
                    context_map=sample_texts["context_map"],
                    text_original=sample_texts["original"],
                    text_amendment=sample_texts["amendment"],
                    langfuse=mock_langfuse,
                )

        assert "no pudo estructurar" in str(exc_info.value)


class TestExtractionGenericError:
    """Errores genéricos (timeout, auth, red) durante la invocación → ExtractionError."""

    def test_runtime_error_raises_extraction_error(self, sample_texts, mock_langfuse):
        def _raise_runtime_error(_):
            raise RuntimeError("Read timeout after 30s")

        mock_llm = _make_mock_llm_with_structured_output(_raise_runtime_error)

        with patch("src.agents.extraction_agent.ChatOpenAI", return_value=mock_llm):
            with pytest.raises(ExtractionError) as exc_info:
                run_extraction_agent(
                    context_map=sample_texts["context_map"],
                    text_original=sample_texts["original"],
                    text_amendment=sample_texts["amendment"],
                    langfuse=mock_langfuse,
                )

        assert "RuntimeError" in str(exc_info.value)


class TestExtractionPydanticValidationError:
    """El dict {include_raw=True} contiene parsing_error → ExtractionError con schema."""

    def test_parsing_error_in_dict_raises_extraction_error(
        self, sample_texts, mock_langfuse
    ):
        """
        Simula el caso donde with_structured_output(include_raw=True) devuelve
        un dict con parsing_error != None (validación Pydantic fallida).
        """
        pydantic_error = ValueError("sections_changed field required")

        def _return_parsing_error_dict(_):
            return {
                "parsed": None,
                "raw": MagicMock(),
                "parsing_error": pydantic_error,
            }

        mock_llm = _make_mock_llm_with_structured_output(_return_parsing_error_dict)

        with patch("src.agents.extraction_agent.ChatOpenAI", return_value=mock_llm):
            with pytest.raises(ExtractionError) as exc_info:
                run_extraction_agent(
                    context_map=sample_texts["context_map"],
                    text_original=sample_texts["original"],
                    text_amendment=sample_texts["amendment"],
                    langfuse=mock_langfuse,
                )

        assert "schema Pydantic" in str(exc_info.value)


class TestExtractionNullResult:
    """El dict {include_raw=True} no tiene parsing_error pero parsed=None → ExtractionError."""

    def test_null_parsed_result_raises_extraction_error(
        self, sample_texts, mock_langfuse
    ):
        """
        Simula el caso defensivo: no hay parsing_error pero el resultado
        parseado es None igualmente (situación inesperada).
        """

        def _return_null_parsed(_):
            return {
                "parsed": None,
                "raw": MagicMock(),
                "parsing_error": None,
            }

        mock_llm = _make_mock_llm_with_structured_output(_return_null_parsed)

        with patch("src.agents.extraction_agent.ChatOpenAI", return_value=mock_llm):
            with pytest.raises(ExtractionError) as exc_info:
                run_extraction_agent(
                    context_map=sample_texts["context_map"],
                    text_original=sample_texts["original"],
                    text_amendment=sample_texts["amendment"],
                    langfuse=mock_langfuse,
                )

        assert "nulo" in str(exc_info.value)

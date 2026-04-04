"""
test_contextualization_agent.py
--------------------------------
Verifica que run_contextualization_agent() lanza ContextualizationError
en todos los escenarios de fallo esperados.

Estrategia de mocking:
  - 'src.agents.contextualization_agent.ChatOpenAI': el LLM se instancia
    dentro de la función, por lo que patcheamos la clase en el módulo destino.
  - En lugar de un MagicMock puro (que no es un Runnable LangChain), usamos
    RunnableLambda para que `prompt | llm` sea una composición válida de LCEL.
  - 'langfuse': fixture de conftest.py.

No se realizan llamadas reales a la API de OpenAI.
"""

import pytest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException

from src.agents.contextualization_agent import run_contextualization_agent
from src.exceptions import ContextualizationError


class TestContextualizationOutputParserException:
    """OutputParserException de LangChain se convierte en ContextualizationError."""

    def test_output_parser_exception_raises_contextualization_error(
        self, sample_texts, mock_langfuse
    ):
        def _raise_output_parser_exception(_):
            raise OutputParserException("El modelo devolvió JSON inválido.")

        with patch(
            "src.agents.contextualization_agent.ChatOpenAI",
            return_value=RunnableLambda(_raise_output_parser_exception),
        ):
            with pytest.raises(ContextualizationError) as exc_info:
                run_contextualization_agent(
                    text_original=sample_texts["original"],
                    text_amendment=sample_texts["amendment"],
                    langfuse=mock_langfuse,
                )

        assert "no pudo procesar" in str(exc_info.value)


class TestContextualizationGenericError:
    """Errores genéricos (timeout, auth, red) se convierten en ContextualizationError."""

    def test_runtime_error_raises_contextualization_error(
        self, sample_texts, mock_langfuse
    ):
        """Connection reset o timeout → ContextualizationError."""

        def _raise_runtime_error(_):
            raise RuntimeError("Connection reset by peer")

        with patch(
            "src.agents.contextualization_agent.ChatOpenAI",
            return_value=RunnableLambda(_raise_runtime_error),
        ):
            with pytest.raises(ContextualizationError) as exc_info:
                run_contextualization_agent(
                    text_original=sample_texts["original"],
                    text_amendment=sample_texts["amendment"],
                    langfuse=mock_langfuse,
                )

        # El mensaje debe incluir el tipo y causa del error
        assert "RuntimeError" in str(exc_info.value)

    def test_openai_auth_error_raises_contextualization_error(
        self, sample_texts, mock_langfuse
    ):
        """Error de auth de OpenAI via LangChain → ContextualizationError."""

        def _raise_auth_error(_):
            raise Exception("AuthenticationError: Invalid API key")

        with patch(
            "src.agents.contextualization_agent.ChatOpenAI",
            return_value=RunnableLambda(_raise_auth_error),
        ):
            with pytest.raises(ContextualizationError):
                run_contextualization_agent(
                    text_original=sample_texts["original"],
                    text_amendment=sample_texts["amendment"],
                    langfuse=mock_langfuse,
                )

"""
test_image_parser.py
---------------------
Verifica que parse_contract_image() lanza ImageParsingError
en todos los escenarios de fallo esperados.

Estrategia de mocking:
  - 'src.image_parser.client': el cliente OpenAI está instanciado a nivel
    de módulo en image_parser.py, por lo que es la target correcta del patch.
  - 'langfuse': fixture de conftest.py (MagicMock compatible con context manager).

No se realizan llamadas reales a la API de OpenAI.
"""

import pytest
from unittest.mock import MagicMock, patch

from openai import OpenAIError

from src.image_parser import parse_contract_image
from src.exceptions import ImageParsingError


class TestImageParserOpenAIError:
    """El cliente de OpenAI lanza OpenAIError durante la llamada a la API Vision."""

    def test_openai_auth_error_raises_image_parsing_error(self, tmp_jpeg, mock_langfuse):
        """AuthenticationError de OpenAI se convierte en ImageParsingError."""
        with patch("src.image_parser.client") as mock_client:
            mock_client.responses.create.side_effect = OpenAIError(
                "Incorrect API key provided."
            )

            with pytest.raises(ImageParsingError) as exc_info:
                parse_contract_image(tmp_jpeg, mock_langfuse, "test_span")

        assert "contrato.jpg" in str(exc_info.value)

    def test_openai_rate_limit_error_raises_image_parsing_error(self, tmp_jpeg, mock_langfuse):
        """Rate limit / timeout de OpenAI se convierte en ImageParsingError."""
        with patch("src.image_parser.client") as mock_client:
            mock_client.responses.create.side_effect = OpenAIError(
                "Rate limit exceeded. Please retry."
            )

            with pytest.raises(ImageParsingError):
                parse_contract_image(tmp_jpeg, mock_langfuse, "test_span")


class TestImageParserUnexpectedError:
    """Errores inesperados (no OpenAIError) también se convierten en ImageParsingError."""

    def test_runtime_error_raises_image_parsing_error(self, tmp_jpeg, mock_langfuse):
        """Un RuntimeError genérico se captura y re-lanza como ImageParsingError."""
        with patch("src.image_parser.client") as mock_client:
            mock_client.responses.create.side_effect = RuntimeError(
                "Unexpected encoding failure"
            )

            with pytest.raises(ImageParsingError):
                parse_contract_image(tmp_jpeg, mock_langfuse, "test_span")


class TestImageParserEmptyResponse:
    """El modelo devuelve texto vacío o una respuesta de rechazo."""

    def _make_mock_response(self, output_text: str) -> MagicMock:
        response = MagicMock()
        response.output_text = output_text
        response.usage = None
        return response

    def test_empty_string_response_raises_image_parsing_error(self, tmp_jpeg, mock_langfuse):
        """output_text vacío indica que el modelo no pudo extraer texto."""
        with patch("src.image_parser.client") as mock_client:
            mock_client.responses.create.return_value = self._make_mock_response("")

            with pytest.raises(ImageParsingError) as exc_info:
                parse_contract_image(tmp_jpeg, mock_langfuse, "test_span")

        assert "no pudo extraer texto" in str(exc_info.value)

    def test_whitespace_only_response_raises_image_parsing_error(self, tmp_jpeg, mock_langfuse):
        """output_text con solo espacios se trata como vacío."""
        with patch("src.image_parser.client") as mock_client:
            mock_client.responses.create.return_value = self._make_mock_response("   \n\t  ")

            with pytest.raises(ImageParsingError):
                parse_contract_image(tmp_jpeg, mock_langfuse, "test_span")

    def test_refusal_response_raises_image_parsing_error(self, tmp_jpeg, mock_langfuse):
        """La cadena de rechazo estándar del modelo lanza ImageParsingError."""
        with patch("src.image_parser.client") as mock_client:
            mock_client.responses.create.return_value = self._make_mock_response(
                "Lo siento, no puedo procesar esta solicitud."
            )

            with pytest.raises(ImageParsingError):
                parse_contract_image(tmp_jpeg, mock_langfuse, "test_span")


class TestImageParserPDFBranch:
    """Verifica que la rama PDF también lanza ImageParsingError correctamente."""

    def test_pdf_openai_error_raises_image_parsing_error(self, tmp_pdf, mock_langfuse):
        """OpenAIError en rama PDF también se convierte en ImageParsingError."""
        with patch("src.image_parser.client") as mock_client:
            mock_client.responses.create.side_effect = OpenAIError("PDF too large")

            with pytest.raises(ImageParsingError):
                parse_contract_image(tmp_pdf, mock_langfuse, "test_pdf_span")

"""
test_api_integration.py
------------------------
Tests de integración para la API REST (FastAPI).

Cubren el ciclo completo request → response de cada endpoint, mockeando
únicamente run_pipeline y save_output_files para evitar llamadas reales a OpenAI.

Se usa FastAPI TestClient (basado en httpx/starlette), que no requiere
levantar un servidor real.

Mocking targets:
    src.api.run_pipeline         ← importado en api.py desde pipeline.py
    src.api.save_output_files    ← ídem
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api import app
from src.models import ContractChangeOutput
from src.exceptions import (
    ImageParsingError,
    ContextualizationError,
    ExtractionError,
    OutputSaveError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient reutilizable para todos los tests del módulo."""
    return TestClient(app)


@pytest.fixture
def mock_result():
    """ContractChangeOutput de ejemplo para los mocks del pipeline."""
    return ContractChangeOutput(
        sections_changed=["Cláusula 1 — Duración", "Cláusula 3 — Precio"],
        topics_touched=["plazos", "precio"],
        summary_of_the_change=(
            "1. **MODIFICACIÓN:** La duración se extendió de 12 a 24 meses.\n"
            "2. **MODIFICACIÓN:** El precio anual subió de USD 10.000 a USD 15.000."
        ),
    )


@pytest.fixture
def pipeline_success(mock_result):
    """Patch de run_pipeline que devuelve un resultado exitoso."""
    with patch("src.api.run_pipeline") as mock:
        mock.return_value = (mock_result, "texto original", "texto enmienda")
        yield mock


@pytest.fixture
def save_success():
    """Patch de save_output_files que simula guardado exitoso."""
    with patch("src.api.save_output_files") as mock:
        mock.return_value = {
            "text_original":  "output/doc_original_extracted.txt",
            "text_amendment": "output/doc_enmienda_extracted.txt",
            "result":         "output/doc_result.json",
        }
        yield mock


def jpeg_files():
    """Multipart form-data con dos JPEGs de prueba."""
    return {
        "original":  ("contrato_original.jpg",  b"\xff\xd8\xff" + b"fake", "image/jpeg"),
        "amendment": ("contrato_enmienda.jpg",   b"\xff\xd8\xff" + b"fake", "image/jpeg"),
    }


def pdf_files():
    """Multipart form-data con dos PDFs de prueba."""
    return {
        "original":  ("contrato_original.pdf",  b"%PDF-1.4fake", "application/pdf"),
        "amendment": ("contrato_enmienda.pdf",   b"%PDF-1.4fake", "application/pdf"),
    }


# ---------------------------------------------------------------------------
# Endpoints informativos
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_contains_service_name(self, client):
        r = client.get("/")
        assert "name" in r.json()
        assert "Contratos" in r.json()["name"]

    def test_lists_analyze_endpoint(self, client):
        r = client.get("/")
        assert "/api/v1/analyze" in r.json()["endpoints"]["analyze"]


class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_is_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — casos de éxito
# ---------------------------------------------------------------------------

class TestAnalyzeSuccess:
    def test_jpeg_returns_200(self, client, pipeline_success, save_success):
        r = client.post("/api/v1/analyze", files=jpeg_files())
        assert r.status_code == 200

    def test_pdf_returns_200(self, client, pipeline_success, save_success):
        r = client.post("/api/v1/analyze", files=pdf_files())
        assert r.status_code == 200

    def test_response_contains_sections_changed(self, client, pipeline_success, save_success):
        r = client.post("/api/v1/analyze", files=jpeg_files())
        body = r.json()
        assert "sections_changed" in body
        assert isinstance(body["sections_changed"], list)
        assert len(body["sections_changed"]) > 0

    def test_response_contains_topics_touched(self, client, pipeline_success, save_success):
        r = client.post("/api/v1/analyze", files=jpeg_files())
        assert "topics_touched" in r.json()

    def test_response_contains_summary(self, client, pipeline_success, save_success):
        r = client.post("/api/v1/analyze", files=jpeg_files())
        assert "summary_of_the_change" in r.json()
        assert len(r.json()["summary_of_the_change"]) > 0


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — query param save_files
# ---------------------------------------------------------------------------

class TestAnalyzeSaveFiles:
    def test_save_files_true_calls_save_output_files(
        self, client, pipeline_success, save_success
    ):
        client.post("/api/v1/analyze?save_files=true", files=jpeg_files())
        save_success.assert_called_once()

    def test_save_files_false_does_not_call_save_output_files(
        self, client, pipeline_success
    ):
        with patch("src.api.save_output_files") as mock_save:
            client.post("/api/v1/analyze?save_files=false", files=jpeg_files())
            mock_save.assert_not_called()

    def test_save_files_defaults_to_true(self, client, pipeline_success, save_success):
        """Sin el query param, save_files debe ser True por defecto."""
        client.post("/api/v1/analyze", files=jpeg_files())
        save_success.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — validación de entrada (422)
# ---------------------------------------------------------------------------

class TestAnalyzeValidation:
    def test_unsupported_extension_returns_422(self, client):
        r = client.post(
            "/api/v1/analyze",
            files={
                "original":  ("contrato.txt", b"texto", "text/plain"),
                "amendment": ("enmienda.jpg", b"\xff\xd8\xff" + b"fake", "image/jpeg"),
            },
        )
        assert r.status_code == 422

    def test_unsupported_extension_error_message(self, client):
        r = client.post(
            "/api/v1/analyze",
            files={
                "original":  ("contrato.txt", b"texto", "text/plain"),
                "amendment": ("enmienda.jpg", b"\xff\xd8\xff" + b"fake", "image/jpeg"),
            },
        )
        detail = r.json()["detail"]
        assert ".txt" in detail or "extensión" in detail.lower() or "JPEG" in detail

    def test_missing_original_field_returns_422(self, client):
        r = client.post(
            "/api/v1/analyze",
            files={"amendment": ("enmienda.jpg", b"\xff\xd8\xff" + b"fake", "image/jpeg")},
        )
        assert r.status_code == 422

    def test_missing_amendment_field_returns_422(self, client):
        r = client.post(
            "/api/v1/analyze",
            files={"original": ("original.jpg", b"\xff\xd8\xff" + b"fake", "image/jpeg")},
        )
        assert r.status_code == 422

    def test_no_files_returns_422(self, client):
        r = client.post("/api/v1/analyze")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/analyze — errores del pipeline → respuestas HTTP tipadas
# ---------------------------------------------------------------------------

class TestAnalyzePipelineErrors:
    def test_image_parsing_error_returns_422(self, client):
        with patch("src.api.run_pipeline", side_effect=ImageParsingError("GPT-4o falló")):
            r = client.post("/api/v1/analyze", files=jpeg_files())
        assert r.status_code == 422

    def test_image_parsing_error_detail_structure(self, client):
        with patch("src.api.run_pipeline", side_effect=ImageParsingError("GPT-4o falló")):
            r = client.post("/api/v1/analyze", files=jpeg_files())
        detail = r.json()["detail"]
        assert detail["error"] == "ImageParsingError"
        assert detail["stage"] == "image_parsing"

    def test_contextualization_error_returns_500(self, client):
        with patch(
            "src.api.run_pipeline",
            side_effect=ContextualizationError("Agente 1 falló"),
        ):
            r = client.post("/api/v1/analyze", files=jpeg_files())
        assert r.status_code == 500

    def test_contextualization_error_detail_structure(self, client):
        with patch(
            "src.api.run_pipeline",
            side_effect=ContextualizationError("Agente 1 falló"),
        ):
            r = client.post("/api/v1/analyze", files=jpeg_files())
        detail = r.json()["detail"]
        assert detail["error"] == "ContextualizationError"
        assert detail["stage"] == "contextualization_agent"

    def test_extraction_error_returns_500(self, client):
        with patch("src.api.run_pipeline", side_effect=ExtractionError("Agente 2 falló")):
            r = client.post("/api/v1/analyze", files=jpeg_files())
        assert r.status_code == 500

    def test_extraction_error_detail_structure(self, client):
        with patch("src.api.run_pipeline", side_effect=ExtractionError("Agente 2 falló")):
            r = client.post("/api/v1/analyze", files=jpeg_files())
        detail = r.json()["detail"]
        assert detail["error"] == "ExtractionError"
        assert detail["stage"] == "extraction_agent"

    def test_output_save_error_returns_200(self, client, mock_result):
        """
        OutputSaveError NO debe interrumpir la respuesta:
        el resultado del pipeline ya fue calculado y se retorna de todas formas.
        """
        with patch("src.api.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = (mock_result, "orig", "amend")
            with patch(
                "src.api.save_output_files",
                side_effect=OutputSaveError("Disco lleno"),
            ):
                r = client.post("/api/v1/analyze", files=jpeg_files())
        assert r.status_code == 200
        assert "sections_changed" in r.json()

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
# Endpoint: GET /api/v1/contracts
# ---------------------------------------------------------------------------

class TestContractsEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/v1/contracts")
        assert r.status_code == 200

    def test_response_has_directory_field(self, client):
        r = client.get("/api/v1/contracts")
        assert "directory" in r.json()
        assert r.json()["directory"] == "data/test_contracts"

    def test_response_has_pairs_field(self, client):
        r = client.get("/api/v1/contracts")
        assert "pairs" in r.json()
        assert isinstance(r.json()["pairs"], list)

    def test_response_has_total_pairs_field(self, client):
        r = client.get("/api/v1/contracts")
        body = r.json()
        assert "total_pairs" in body
        assert body["total_pairs"] == len(body["pairs"])

    def test_response_has_individual_files_field(self, client):
        r = client.get("/api/v1/contracts")
        assert "individual_files" in r.json()
        assert isinstance(r.json()["individual_files"], list)

    def test_pairs_have_correct_structure(self, client):
        """Cada par debe tener 'pair', 'original' y 'amendment'."""
        r = client.get("/api/v1/contracts")
        for pair in r.json()["pairs"]:
            assert "pair"      in pair
            assert "original"  in pair
            assert "amendment" in pair

    def test_pairs_files_are_valid_extensions(self, client):
        """Los archivos en cada par deben ser JPEG, PNG o PDF."""
        valid_exts = {".jpg", ".jpeg", ".png", ".pdf"}
        r = client.get("/api/v1/contracts")
        for pair in r.json()["pairs"]:
            import os
            assert os.path.splitext(pair["original"])[1].lower()  in valid_exts
            assert os.path.splitext(pair["amendment"])[1].lower() in valid_exts

    def test_no_server_needed(self, client):
        """El endpoint no requiere archivos subidos ni autenticación."""
        r = client.get("/api/v1/contracts")
        assert r.status_code == 200  # siempre responde, incluso si el dir está vacío


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
        """Upload endpoint requires both files — FastAPI enforces this at schema level."""
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
        """No files at all → 422 (FastAPI enforces required fields)."""
        r = client.post("/api/v1/analyze")
        assert r.status_code == 422

    def test_sample_no_pair_param_returns_422(self, client):
        """Sample endpoint requires the pair query param."""
        r = client.post("/api/v1/analyze/sample")
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

    def test_bad_contracts_error_returns_422(self, client):
        """Contratos no comparables → 422 con error BadContractsError."""
        from src.exceptions import BadContractsError
        with patch("src.api.run_pipeline", side_effect=BadContractsError("Los contratos no son comparables")):
            r = client.post("/api/v1/analyze", files=jpeg_files())
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "BadContractsError"


# ---------------------------------------------------------------------------
# POST /api/v1/analyze/sample
# ---------------------------------------------------------------------------

class TestAnalyzeSampleEndpoint:
    def test_valid_pair_returns_200(self, client, pipeline_success, save_success):
        """par=documento_1 con archivos reales en test_contracts → 200."""
        r = client.post("/api/v1/analyze/sample?pair=documento_1&save_files=false")
        assert r.status_code == 200

    def test_valid_pair_returns_sections_changed(self, client, pipeline_success, save_success):
        r = client.post("/api/v1/analyze/sample?pair=documento_1&save_files=false")
        assert "sections_changed" in r.json()

    def test_valid_pair_pipeline_called_with_test_contracts_paths(
        self, client, save_success, mock_result
    ):
        """El pipeline recibe paths dentro de data/test_contracts/."""
        with patch("src.api.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = (mock_result, "orig", "amend")
            client.post("/api/v1/analyze/sample?pair=documento_1&save_files=false")

        call_args = mock_pipeline.call_args
        assert call_args is not None
        orig_path  = call_args.kwargs.get("original_path")  or call_args.args[0]
        amend_path = call_args.kwargs.get("amendment_path") or call_args.args[1]
        assert "test_contracts" in orig_path
        assert "test_contracts" in amend_path

    def test_invalid_pair_returns_422(self, client):
        """Un par que no existe devuelve 422."""
        r = client.post("/api/v1/analyze/sample?pair=no_existe_este_par")
        assert r.status_code == 422

    def test_invalid_pair_detail_has_available_pairs(self, client):
        r = client.post("/api/v1/analyze/sample?pair=no_existe_este_par")
        detail = r.json()["detail"]
        assert "available_pairs" in detail
        assert isinstance(detail["available_pairs"], list)

    def test_missing_pair_param_returns_422(self, client):
        """El parámetro pair es obligatorio — FastAPI lo enforza."""
        r = client.post("/api/v1/analyze/sample")
        assert r.status_code == 422

    def test_save_files_false_does_not_call_save(self, client, mock_result):
        with patch("src.api.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = (mock_result, "orig", "amend")
            with patch("src.api.save_output_files") as mock_save:
                client.post("/api/v1/analyze/sample?pair=documento_1&save_files=false")
                mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# GET /api/v1/contracts/{filename}
# ---------------------------------------------------------------------------

class TestContractFileEndpoint:
    # --- Modo raw (default) ------------------------------------------------

    def test_raw_jpeg_returns_200(self, client):
        r = client.get("/api/v1/contracts/documento_1__original.jpg")
        assert r.status_code == 200

    def test_raw_jpeg_content_type(self, client):
        r = client.get("/api/v1/contracts/documento_1__original.jpg?mode=raw")
        assert "image/jpeg" in r.headers["content-type"]

    def test_raw_pdf_returns_200(self, client):
        r = client.get("/api/v1/contracts/contrato_alquiler__original.pdf?mode=raw")
        assert r.status_code == 200

    def test_raw_pdf_content_type(self, client):
        r = client.get("/api/v1/contracts/contrato_alquiler__original.pdf?mode=raw")
        assert "pdf" in r.headers["content-type"]

    def test_raw_is_default_mode(self, client):
        """Sin ?mode, el comportamiento debe ser raw (devuelve bytes, no JSON)."""
        r = client.get("/api/v1/contracts/documento_1__original.jpg")
        assert r.status_code == 200
        assert "image" in r.headers["content-type"]

    def test_raw_response_is_binary(self, client):
        r = client.get("/api/v1/contracts/documento_1__original.jpg")
        assert len(r.content) > 0

    # --- Modo text (cache) ------------------------------------------------

    def test_text_cached_file_returns_200(self, client):
        """documento_1__original.jpg tiene texto cacheado."""
        r = client.get("/api/v1/contracts/documento_1__original.jpg?mode=text")
        assert r.status_code == 200

    def test_text_cached_source_is_cache(self, client):
        r = client.get("/api/v1/contracts/documento_1__original.jpg?mode=text")
        assert r.json()["source"] == "cache"

    def test_text_cached_has_nonempty_text(self, client):
        r = client.get("/api/v1/contracts/documento_1__original.jpg?mode=text")
        body = r.json()
        assert body["text"] is not None
        assert len(body["text"]) > 10

    def test_text_no_cache_returns_200(self, client):
        """documento_2__enmienda.jpg no tiene texto cacheado."""
        r = client.get("/api/v1/contracts/documento_2__enmienda.jpg?mode=text")
        assert r.status_code == 200

    def test_text_no_cache_source_is_not_available(self, client):
        r = client.get("/api/v1/contracts/documento_2__enmienda.jpg?mode=text")
        body = r.json()
        assert body["source"] == "not_available"
        assert body["text"] is None

    def test_text_no_cache_has_hint(self, client):
        r = client.get("/api/v1/contracts/documento_2__enmienda.jpg?mode=text")
        assert "hint" in r.json()

    # --- Validaciones / errores -------------------------------------------

    def test_nonexistent_file_returns_404(self, client):
        r = client.get("/api/v1/contracts/no_existe.jpg")
        assert r.status_code == 404

    def test_invalid_extension_returns_422(self, client):
        r = client.get("/api/v1/contracts/contrato.txt")
        assert r.status_code == 422

    def test_path_traversal_encoded_returns_error(self, client):
        """Intento de path traversal con URL encoding debe ser rechazado."""
        r = client.get("/api/v1/contracts/..%2Fsrc%2Fapi.py")
        # La extensión .py falla la validación → 422
        assert r.status_code in (404, 422)

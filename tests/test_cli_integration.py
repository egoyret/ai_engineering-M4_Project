"""
test_cli_integration.py
------------------------
Tests de integración para el CLI (src/main.py).

Cubren el comportamiento completo de main(): rutas resueltas, exit codes,
mensajes de error y output en consola, mockeando run_pipeline y
save_output_files para evitar llamadas reales a OpenAI.

Técnicas utilizadas:
    - pytest.raises(SystemExit) + .value.code para verificar exit codes.
    - capsys para capturar stdout/stderr.
    - patch("sys.argv", [...]) para simular argumentos de línea de comandos.
    - tmp_path para crear archivos de prueba temporales.
    - patch("src.main.run_pipeline") y patch("src.main.save_output_files")
      para interceptar las llamadas al pipeline.
"""

import sys
import pytest
from unittest.mock import patch

from src.main import main
from src.models import ContractChangeOutput
from src.exceptions import (
    ImageParsingError,
    ContextualizationError,
    ExtractionError,
    OutputSaveError,
    ContractPipelineError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_result():
    return ContractChangeOutput(
        sections_changed=["Cláusula 1"],
        topics_touched=["precio"],
        summary_of_the_change="1. **MODIFICACIÓN:** El precio subió de 10 a 15.",
    )


@pytest.fixture
def contract_files(tmp_path):
    """Par de archivos de contrato temporales que existen en disco."""
    original  = tmp_path / "doc_original.jpg"
    amendment = tmp_path / "doc_enmienda.jpg"
    original.write_bytes(b"\xff\xd8\xff" + b"fake-jpeg")
    amendment.write_bytes(b"\xff\xd8\xff" + b"fake-jpeg")
    return str(original), str(amendment)


@pytest.fixture
def pipeline_success(mock_result):
    """Patch de run_pipeline que devuelve éxito."""
    with patch("src.main.run_pipeline") as mock:
        mock.return_value = (mock_result, "texto original", "texto enmienda")
        yield mock


@pytest.fixture
def save_success():
    """Patch de save_output_files que simula guardado exitoso."""
    with patch("src.main.save_output_files") as mock:
        mock.return_value = {
            "text_original":  "output/doc_original_extracted.txt",
            "text_amendment": "output/doc_enmienda_extracted.txt",
            "result":         "output/doc_result.json",
        }
        yield mock


# ---------------------------------------------------------------------------
# Casos de éxito
# ---------------------------------------------------------------------------

class TestCLISuccess:
    def test_custom_paths_runs_pipeline(
        self, contract_files, pipeline_success, save_success
    ):
        """Con rutas válidas en argv, el pipeline se ejecuta exactamente una vez."""
        original, amendment = contract_files
        with patch("sys.argv", ["main.py", original, amendment]):
            main()

        pipeline_success.assert_called_once_with(original, amendment)

    def test_success_prints_result(
        self, contract_files, pipeline_success, save_success, capsys
    ):
        """El JSON del resultado aparece en stdout."""
        original, amendment = contract_files
        with patch("sys.argv", ["main.py", original, amendment]):
            main()

        stdout = capsys.readouterr().out
        assert "sections_changed" in stdout
        assert "Cláusula 1" in stdout

    def test_success_prints_pipeline_completed(
        self, contract_files, pipeline_success, save_success, capsys
    ):
        """El mensaje de finalización exitosa aparece en stdout."""
        original, amendment = contract_files
        with patch("sys.argv", ["main.py", original, amendment]):
            main()

        assert "Pipeline completado exitosamente" in capsys.readouterr().out

    def test_success_prints_saved_filenames(
        self, contract_files, pipeline_success, save_success, capsys
    ):
        """Los nombres de los archivos guardados aparecen en stdout."""
        original, amendment = contract_files
        with patch("sys.argv", ["main.py", original, amendment]):
            main()

        stdout = capsys.readouterr().out
        assert "doc_original_extracted.txt" in stdout
        assert "doc_enmienda_extracted.txt" in stdout
        assert "doc_result.json" in stdout


# ---------------------------------------------------------------------------
# Validación de archivos (antes de llamar al pipeline)
# ---------------------------------------------------------------------------

class TestCLIFileValidation:
    def test_missing_original_exits_1(self, tmp_path, capsys):
        """Si el original no existe, el CLI termina con exit code 1."""
        missing  = str(tmp_path / "nonexistent.jpg")
        existing = tmp_path / "enmienda.jpg"
        existing.write_bytes(b"fake")

        with patch("sys.argv", ["main.py", missing, str(existing)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        assert "no se encontró" in capsys.readouterr().out

    def test_missing_amendment_exits_1(self, tmp_path, capsys):
        """Si la enmienda no existe, el CLI termina con exit code 1."""
        existing = tmp_path / "original.jpg"
        existing.write_bytes(b"fake")
        missing  = str(tmp_path / "nonexistent.jpg")

        with patch("sys.argv", ["main.py", str(existing), missing]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_missing_file_prints_format_hint(self, tmp_path, capsys):
        """El mensaje de error menciona los formatos aceptados."""
        missing = str(tmp_path / "nonexistent.jpg")
        with patch("sys.argv", ["main.py", missing, missing]):
            with pytest.raises(SystemExit):
                main()

        stdout = capsys.readouterr().out
        assert "JPEG" in stdout or "PDF" in stdout


# ---------------------------------------------------------------------------
# Errores del pipeline → exit code 1 + mensajes claros
# ---------------------------------------------------------------------------

class TestCLIPipelineErrors:
    def test_image_parsing_error_exits_1(self, contract_files, capsys):
        original, amendment = contract_files
        with patch("src.main.run_pipeline", side_effect=ImageParsingError("Vision falló")):
            with patch("sys.argv", ["main.py", original, amendment]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_image_parsing_error_prints_stage(self, contract_files, capsys):
        original, amendment = contract_files
        with patch("src.main.run_pipeline", side_effect=ImageParsingError("Vision falló")):
            with patch("sys.argv", ["main.py", original, amendment]):
                with pytest.raises(SystemExit):
                    main()

        assert "PARSING" in capsys.readouterr().out

    def test_contextualization_error_exits_1(self, contract_files, capsys):
        original, amendment = contract_files
        with patch(
            "src.main.run_pipeline",
            side_effect=ContextualizationError("Agente 1 falló"),
        ):
            with patch("sys.argv", ["main.py", original, amendment]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        assert "CONTEXTUALIZACIÓN" in capsys.readouterr().out

    def test_extraction_error_exits_1(self, contract_files, capsys):
        original, amendment = contract_files
        with patch("src.main.run_pipeline", side_effect=ExtractionError("Agente 2 falló")):
            with patch("sys.argv", ["main.py", original, amendment]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        assert "EXTRACCIÓN" in capsys.readouterr().out

    def test_generic_pipeline_error_exits_1(self, contract_files, capsys):
        """ContractPipelineError genérico (no subtipo) también termina con exit 1."""
        original, amendment = contract_files
        with patch(
            "src.main.run_pipeline",
            side_effect=ContractPipelineError("Error desconocido del pipeline"),
        ):
            with patch("sys.argv", ["main.py", original, amendment]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_unexpected_exception_is_re_raised(self, contract_files):
        """Excepciones no esperadas (no ContractPipelineError) se re-lanzan (no sys.exit)."""
        original, amendment = contract_files
        with patch("src.main.run_pipeline", side_effect=RuntimeError("internal crash")):
            with patch("sys.argv", ["main.py", original, amendment]):
                with pytest.raises(RuntimeError):
                    main()


# ---------------------------------------------------------------------------
# Error en guardado de archivos
# ---------------------------------------------------------------------------

class TestCLIOutputSaveError:
    def test_output_save_error_exits_1(
        self, contract_files, pipeline_success, capsys
    ):
        """Si save_output_files falla, el CLI termina con exit code 1."""
        original, amendment = contract_files
        with patch(
            "src.main.save_output_files",
            side_effect=OutputSaveError("Permiso denegado en output/"),
        ):
            with patch("sys.argv", ["main.py", original, amendment]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

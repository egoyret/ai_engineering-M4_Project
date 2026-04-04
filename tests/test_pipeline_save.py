"""
test_pipeline_save.py
----------------------
Verifica que save_output_files() lanza OutputSaveError
cuando hay errores de I/O al escribir en disco.

Estrategia de mocking:
  - 'os.makedirs': patcheado para simular fallo de permisos en el directorio.
  - 'builtins.open': patcheado para simular disco lleno o permiso denegado
    durante la escritura de los archivos de texto.
  - ContractChangeOutput: MagicMock con model_dump() configurado.

No se escribe en disco real durante estos tests.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open

from src.pipeline import save_output_files
from src.exceptions import OutputSaveError


@pytest.fixture
def mock_result():
    """Mock de ContractChangeOutput con model_dump() configurado."""
    result = MagicMock()
    result.model_dump.return_value = {
        "sections_changed": ["Cláusula 1"],
        "topics_touched": ["precio"],
        "summary_of_the_change": "Se modificó el precio.",
    }
    return result


@pytest.fixture
def file_args(tmp_path, mock_result):
    """Argumentos comunes para save_output_files."""
    return {
        "text_original": "Texto del contrato original.",
        "text_amendment": "Texto de la enmienda.",
        "result": mock_result,
        "original_path": str(tmp_path / "documento_1__original.jpg"),
        "amendment_path": str(tmp_path / "documento_1__enmienda.jpg"),
        "output_dir": str(tmp_path / "output"),
    }


class TestOutputSaveErrorOnMakeDirs:
    """os.makedirs lanza OSError → OutputSaveError."""

    def test_makedirs_permission_error_raises_output_save_error(self, file_args):
        """Si no hay permisos para crear output/, se lanza OutputSaveError."""
        with patch("src.pipeline.os.makedirs", side_effect=OSError("Permission denied")):
            with pytest.raises(OutputSaveError) as exc_info:
                save_output_files(**file_args)

        assert "No se pudieron guardar" in str(exc_info.value)
        assert "output" in str(exc_info.value).lower()

    def test_makedirs_readonly_filesystem_raises_output_save_error(self, file_args):
        """Sistema de archivos de solo lectura → OutputSaveError."""
        with patch(
            "src.pipeline.os.makedirs",
            side_effect=OSError("Read-only file system"),
        ):
            with pytest.raises(OutputSaveError):
                save_output_files(**file_args)


class TestOutputSaveErrorOnFileWrite:
    """open() lanza OSError durante la escritura → OutputSaveError."""

    def test_open_permission_error_raises_output_save_error(self, file_args):
        """Permiso denegado al abrir el archivo de texto → OutputSaveError."""
        with patch("src.pipeline.os.makedirs"):  # makedirs OK
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                with pytest.raises(OutputSaveError) as exc_info:
                    save_output_files(**file_args)

        assert "No se pudieron guardar" in str(exc_info.value)

    def test_open_disk_full_raises_output_save_error(self, file_args):
        """Disco lleno durante la escritura → OutputSaveError."""
        with patch("src.pipeline.os.makedirs"):
            with patch("builtins.open", side_effect=OSError("No space left on device")):
                with pytest.raises(OutputSaveError):
                    save_output_files(**file_args)

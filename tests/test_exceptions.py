"""
test_exceptions.py
------------------
Verifica la jerarquía de herencia de las excepciones personalizadas.

No se mockea nada: son tests puramente estructurales que validan
que la jerarquía esté definida correctamente en exceptions.py.
"""

import pytest

from src.exceptions import (
    ContractPipelineError,
    ImageParsingError,
    ContextualizationError,
    ExtractionError,
    OutputSaveError,
)


class TestExceptionHierarchy:
    """ContractPipelineError es la base de todas las excepciones del pipeline."""

    def test_contract_pipeline_error_is_exception(self):
        """La excepción base hereda de la excepción estándar de Python."""
        assert issubclass(ContractPipelineError, Exception)

    def test_image_parsing_error_is_pipeline_error(self):
        assert issubclass(ImageParsingError, ContractPipelineError)

    def test_contextualization_error_is_pipeline_error(self):
        assert issubclass(ContextualizationError, ContractPipelineError)

    def test_extraction_error_is_pipeline_error(self):
        assert issubclass(ExtractionError, ContractPipelineError)

    def test_output_save_error_is_pipeline_error(self):
        assert issubclass(OutputSaveError, ContractPipelineError)


class TestExceptionCatching:
    """Valida que el catch genérico por ContractPipelineError atrapa a los subtipos."""

    def test_image_parsing_error_caught_by_base(self):
        with pytest.raises(ContractPipelineError):
            raise ImageParsingError("La imagen no se pudo procesar.")

    def test_contextualization_error_caught_by_base(self):
        with pytest.raises(ContractPipelineError):
            raise ContextualizationError("El agente 1 falló.")

    def test_extraction_error_caught_by_base(self):
        with pytest.raises(ContractPipelineError):
            raise ExtractionError("El agente 2 falló.")

    def test_output_save_error_caught_by_base(self):
        with pytest.raises(ContractPipelineError):
            raise OutputSaveError("No se pudo guardar el archivo.")

    def test_pipeline_error_not_caught_by_value_error(self):
        """Las excepciones del pipeline NO deben ser atrapadas por tipos no relacionados."""
        with pytest.raises(ImageParsingError):
            try:
                raise ImageParsingError("error")
            except ValueError:
                pass  # No debe entrar aquí


class TestExceptionMessages:
    """El mensaje se preserva correctamente en la excepción."""

    def test_image_parsing_error_preserves_message(self):
        msg = "❌ Error de Vision: autenticación fallida."
        err = ImageParsingError(msg)
        assert str(err) == msg

    def test_output_save_error_preserves_message(self):
        msg = "❌ No se pudo guardar en output/."
        err = OutputSaveError(msg)
        assert str(err) == msg

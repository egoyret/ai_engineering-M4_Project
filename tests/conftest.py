"""
conftest.py
-----------
Fixtures compartidas para todos los tests de excepciones.
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_langfuse():
    """
    MagicMock de Langfuse compatible con el context manager
    start_as_current_observation() usado en cada módulo.
    """
    lf = MagicMock()
    # start_as_current_observation es usado como 'with lf.start_as_current_observation(...)'
    # MagicMock soporta __enter__ / __exit__ de forma nativa.
    return lf


@pytest.fixture
def tmp_jpeg(tmp_path):
    """Archivo JPEG temporal con contenido fake para tests de image_parser."""
    img = tmp_path / "contrato.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"fake-jpeg-content")  # magic bytes JPEG
    return str(img)


@pytest.fixture
def tmp_pdf(tmp_path):
    """Archivo PDF temporal con contenido fake para tests de image_parser (rama PDF)."""
    pdf = tmp_path / "contrato.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake-pdf-content")  # magic bytes PDF
    return str(pdf)


@pytest.fixture
def sample_texts():
    """Textos de contratos simples para usar en tests de los agentes."""
    return {
        "original": "CONTRATO DE LICENCIA\n1. Duración: 12 meses.\n2. Precio: USD 10.000.",
        "amendment": "ADENDA AL CONTRATO\n1. Duración: 24 meses.\n2. Precio: USD 15.000.",
        "context_map": "Sección 1 (original → adenda): Duración cambió de 12 a 24 meses.",
    }

"""
api.py
------
API REST del sistema de análisis de contratos.

Expone el pipeline como un endpoint HTTP usando FastAPI.
Ambas interfaces (CLI y API) comparten el mismo código a través de pipeline.py.

Endpoints:
    GET  /                      → Info del servicio
    GET  /health                → Health check
    POST /api/v1/analyze        → Analiza un par de imágenes de contratos

Uso:
    uvicorn src.api:app --reload --port 8000

Swagger UI (generado automáticamente por FastAPI):
    http://localhost:8000/docs
"""

import os
import sys
import tempfile
import logging

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

# Asegurar que la raíz del proyecto está en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline, save_output_files
from src.models import ContractChangeOutput
from src.exceptions import (
    ImageParsingError,
    ContextualizationError,
    ExtractionError,
    OutputSaveError,
    ContractPipelineError,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("contract-analysis-api")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agente Autónomo de Comparación de Contratos",
    description=(
        "Recibe las imágenes escaneadas de un contrato original y su enmienda, "
        "las analiza con IA (GPT-4o Vision + agentes LangChain) y devuelve un "
        "reporte estructurado con los cambios detectados."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_image_file(file: UploadFile, field_name: str) -> None:
    """Valida que el archivo subido sea una imagen JPEG o PNG."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"El campo '{field_name}' debe ser una imagen JPEG o PNG. "
                   f"Extensión recibida: '{ext or 'sin extensión'}'."
        )


async def _save_upload_to_tempfile(upload: UploadFile) -> str:
    """
    Escribe el contenido de un UploadFile en un archivo temporal en disco.

    FastAPI recibe los archivos en memoria. parse_contract_image() necesita
    una ruta en disco, por lo que escribimos un temp file y retornamos su path.
    El caller es responsable de eliminar el archivo temporal después de usarlo.

    Returns:
        Path absoluto al archivo temporal creado.
    """
    ext = os.path.splitext(upload.filename or ".jpg")[1].lower() or ".jpg"
    content = await upload.read()

    # delete=False: necesitamos que el archivo persista mientras lo usa el pipeline
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        return tmp.name


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Información del servicio")
def root():
    """Retorna información básica del servicio y los endpoints disponibles."""
    return {
        "name":        "Agente Autónomo de Comparación de Contratos",
        "version":     "1.0.0",
        "description": "Pipeline de análisis multimodal de contratos con GPT-4o y LangChain.",
        "endpoints": {
            "analyze": "POST /api/v1/analyze",
            "health":  "GET  /health",
            "docs":    "GET  /docs",
        },
    }


@app.get("/health", summary="Health check")
def health():
    """Verifica que el servicio está activo y funcionando."""
    return {"status": "ok", "service": "contract-analysis-api"}


@app.post(
    "/api/v1/analyze",
    response_model=ContractChangeOutput,
    summary="Analizar par de contratos",
    description=(
        "Recibe dos imágenes (contrato original y enmienda) como multipart/form-data, "
        "ejecuta el pipeline completo de análisis y devuelve un JSON estructurado "
        "con las secciones modificadas, temas afectados y el resumen de los cambios.\n\n"
        "**Formatos aceptados**: JPEG, PNG\n\n"
        "**Tiempo estimado**: 30-60 segundos (llamadas a OpenAI)"
    ),
)
async def analyze_contracts(
    original:   UploadFile = File(..., description="Imagen del contrato original (JPEG/PNG)"),
    amendment:  UploadFile = File(..., description="Imagen de la enmienda o adenda (JPEG/PNG)"),
    save_files: bool       = Query(
        default=True,
        description=(
            "Si es true (por defecto), guarda los textos extraídos y el resultado JSON "
            "en la carpeta output/. Si es false, sólo devuelve el JSON sin tocar disco."
        ),
    ),
) -> ContractChangeOutput:
    """
    Endpoint principal: ejecuta el pipeline y devuelve el resultado de la comparación.
    """
    # --- Validar tipos de archivo ---
    _validate_image_file(original,  "original")
    _validate_image_file(amendment, "amendment")

    logger.info(
        "📥 Nueva solicitud — original: %s | amendment: %s | save_files: %s",
        original.filename, amendment.filename, save_files,
    )

    # --- Escribir UploadFiles en archivos temporales ---
    tmp_original_path  = None
    tmp_amendment_path = None

    try:
        tmp_original_path  = await _save_upload_to_tempfile(original)
        tmp_amendment_path = await _save_upload_to_tempfile(amendment)

        logger.info("🔍 Iniciando pipeline...")

        # --- Ejecutar el pipeline core ---
        result, text_original, text_amendment = run_pipeline(
            original_path=tmp_original_path,
            amendment_path=tmp_amendment_path,
        )

        logger.info("✅ Pipeline completado. Secciones modificadas: %s", result.sections_changed)

        # --- Guardar archivos en output/ (si save_files=True) ---
        if save_files:
            # Usamos los nombres originales del upload para el naming de los archivos de salida
            saved_paths = save_output_files(
                text_original=text_original,
                text_amendment=text_amendment,
                result=result,
                # Nombrar con el filename del upload, no el path temporal
                original_path=os.path.join(OUTPUT_DIR, original.filename or "original.jpg"),
                amendment_path=os.path.join(OUTPUT_DIR, amendment.filename or "amendment.jpg"),
                output_dir=OUTPUT_DIR,
            )
            logger.info(
                "💾 Archivos guardados: %s | %s | %s",
                os.path.basename(saved_paths["text_original"]),
                os.path.basename(saved_paths["text_amendment"]),
                os.path.basename(saved_paths["result"]),
            )

        return result

    # --- Traducir excepciones del pipeline a respuestas HTTP ---
    except ImageParsingError as e:
        logger.error("❌ ImageParsingError: %s", e)
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "ImageParsingError",
                "message": str(e),
                "stage":   "image_parsing",
                "hint":    "Verificá que las imágenes sean JPEG/PNG legibles y que contengan contratos.",
            },
        )

    except ContextualizationError as e:
        logger.error("❌ ContextualizationError: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "ContextualizationError",
                "message": str(e),
                "stage":   "contextualization_agent",
                "hint":    "Intentá nuevamente. Si persiste, verificá la cuota de OPENAI_API_KEY.",
            },
        )

    except ExtractionError as e:
        logger.error("❌ ExtractionError: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "ExtractionError",
                "message": str(e),
                "stage":   "extraction_agent",
                "hint":    "El modelo no pudo producir el JSON requerido. Intentá nuevamente.",
            },
        )

    except OutputSaveError as e:
        # El pipeline fue exitoso pero falló el guardado en disco.
        # Retornamos el resultado de todas formas con un warning en el log.
        logger.warning("⚠️  OutputSaveError (resultado disponible de todas formas): %s", e)
        return result

    except ContractPipelineError as e:
        logger.error("❌ ContractPipelineError: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "PipelineError", "message": str(e)},
        )

    except Exception as e:
        logger.exception("🔴 Error inesperado en el pipeline")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   type(e).__name__,
                "message": str(e),
                "hint":    "Error inesperado. Revisá los logs del servidor.",
            },
        )

    finally:
        # --- Siempre eliminar los archivos temporales ---
        for tmp_path in [tmp_original_path, tmp_amendment_path]:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

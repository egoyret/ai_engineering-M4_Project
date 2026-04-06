"""
api.py
------
API REST del sistema de análisis de contratos.

Expone el pipeline como un endpoint HTTP usando FastAPI.
Ambas interfaces (CLI y API) comparten el mismo código a través de pipeline.py.

Endpoints:
    GET  /                              → Info del servicio
    GET  /health                        → Health check
    GET  /api/v1/contracts              → Lista los contratos de ejemplo
    GET  /api/v1/contracts/{filename}   → Ver un contrato individual (raw o texto extraído)
    POST /api/v1/analyze                → Analiza archivos subidos por el usuario
    POST /api/v1/analyze/sample         → Analiza un par de contratos de ejemplo (?pair=)

Uso:
    uvicorn src.api:app --reload --port 8000

Swagger UI (generado automáticamente por FastAPI):
    http://localhost:8000/docs
"""

import os
import sys
import tempfile
import logging

from typing import Optional, Literal

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse

# Asegurar que la raíz del proyecto está en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline, save_output_files
from src.models import ContractChangeOutput
from src.exceptions import (
    ImageParsingError,
    ContextualizationError,
    ExtractionError,
    BadContractsError,
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
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR        = os.path.join(PROJECT_ROOT, "output")
TEST_CONTRACTS_DIR = os.path.join(PROJECT_ROOT, "data", "test_contracts")

ALLOWED_MIME_TYPES  = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".pdf"}

EXTENSION_CONTENT_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".pdf":  "application/pdf",
}

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agente Autónomo de Comparación de Contratos",
    description=(
        "Recibe los documentos de un contrato original y su enmienda (JPEG, PNG o PDF), "
        "los analiza con IA (GPT-4o Vision + agentes LangChain) y devuelve un "
        "reporte estructurado con los cambios detectados."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sample_pair(pair_name: str) -> tuple:
    """
    Resuelve los paths del par de ejemplo solicitado desde data/test_contracts/.

    Returns:
        Tupla (original_path, amendment_path) con rutas absolutas.
    Raises:
        HTTPException 422 si el par no existe o está incompleto.
    """
    if not os.path.isdir(TEST_CONTRACTS_DIR):
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "NoPairsDirectory",
                "message": "El directorio de contratos de ejemplo no existe en este servidor.",
            },
        )

    files = sorted(
        f for f in os.listdir(TEST_CONTRACTS_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    )

    pair_files = [
        f for f in files
        if os.path.splitext(f)[0].split("__")[0] == pair_name
    ]

    if not pair_files:
        available = sorted({
            os.path.splitext(f)[0].split("__")[0] for f in files
        })
        raise HTTPException(
            status_code=422,
            detail={
                "error":           "PairNotFound",
                "message":         f"El par '{pair_name}' no existe en los contratos de ejemplo.",
                "available_pairs": available,
            },
        )

    original_file  = next((f for f in pair_files if "original"  in f.lower()), None)
    amendment_file = next((f for f in pair_files if any(
        kw in f.lower() for kw in ("enmienda", "amendment", "adenda")
    )), None)

    if not original_file or not amendment_file:
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "IncompletePair",
                "message": f"El par '{pair_name}' no tiene ambos archivos (original y enmienda).",
            },
        )

    return (
        os.path.join(TEST_CONTRACTS_DIR, original_file),
        os.path.join(TEST_CONTRACTS_DIR, amendment_file),
    )


def _validate_document_file(file: UploadFile, field_name: str) -> None:
    """Valida que el archivo subido sea JPEG, PNG o PDF."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"El campo '{field_name}' debe ser JPEG, PNG o PDF. "
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
    ext = os.path.splitext(upload.filename or ".jpg")[1].lower() or ".jpg"  # preserva .pdf
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
            "contracts":       "GET  /api/v1/contracts",
            "contract_file":   "GET  /api/v1/contracts/{filename}",
            "analyze":         "POST /api/v1/analyze",
            "analyze_sample":  "POST /api/v1/analyze/sample",
            "health":          "GET  /health",
            "docs":            "GET  /docs",
        },
    }


@app.get("/health", summary="Health check")
def health():
    """Verifica que el servicio está activo y funcionando."""
    return {"status": "ok", "service": "contract-analysis-api"}


@app.get(
    "/api/v1/contracts",
    summary="Listar contratos de ejemplo",
    description=(
        "Lista los archivos de contratos de ejemplo disponibles en `data/test_contracts/`.\n\n"
        "Los archivos se agrupan por pares (original + enmienda) según el prefijo común "
        "de sus nombres. Los nombres de par pueden usarse directamente en "
        "`POST /api/v1/analyze/sample`."
    ),
)
def list_contracts():
    """
    Devuelve los nombres de los contratos de ejemplo, agrupados por par.

    Ejemplo de respuesta:
    {
      "directory": "data/test_contracts",
      "pairs": [
        {
          "pair": "documento_1",
          "original":  "documento_1__original.jpg",
          "amendment": "documento_1__enmienda.jpg"
        },
        ...
      ],
      "individual_files": ["archivo_sin_par.pdf"]
    }
    """
    if not os.path.isdir(TEST_CONTRACTS_DIR):
        return {"directory": "data/test_contracts", "pairs": [], "individual_files": []}

    files = sorted(
        f for f in os.listdir(TEST_CONTRACTS_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    )

    # Agrupar por prefijo común (antes del segundo '__')
    # Ej: "documento_1__original.jpg" → prefijo "documento_1"
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for filename in files:
        stem = os.path.splitext(filename)[0]       # "documento_1__original"
        parts = stem.split("__", 1)                # ["documento_1", "original"]
        prefix = parts[0] if len(parts) > 1 else stem
        groups[prefix].append(filename)

    pairs = []
    individual = []

    for prefix, group_files in sorted(groups.items()):
        # Detectar original y enmienda dentro del grupo
        original_file  = next((f for f in group_files if "original"  in f.lower()), None)
        amendment_file = next((f for f in group_files if any(
            kw in f.lower() for kw in ("enmienda", "amendment", "adenda", "modificado")
        )), None)

        if original_file and amendment_file:
            pairs.append({
                "pair":      prefix,
                "original":  original_file,
                "amendment": amendment_file,
            })
        else:
            individual.extend(group_files)

    return {
        "directory":        "data/test_contracts",
        "total_pairs":      len(pairs),
        "pairs":            pairs,
        "individual_files": individual,
    }


@app.get(
    "/api/v1/contracts/{filename}",
    summary="Ver contrato de ejemplo",
    description=(
        "Devuelve un archivo de contrato de ejemplo de `data/test_contracts/`.\n\n"
        "**Modos de uso** (param `mode`):\n"
        "- `raw` (default): devuelve el archivo original (imagen o PDF) con el Content-Type correcto. "
        "El browser puede renderizarlo directamente.\n"
        "- `text`: devuelve el texto extraído por el pipeline si fue procesado previamente. "
        "Si no hay texto cacheado, retorna `source: not_available` sin llamar a OpenAI."
    ),
)
def get_contract_file(
    filename: str,
    mode: Literal["raw", "text"] = Query(
        default="raw",
        description="'raw' (default): descarga el archivo. 'text': devuelve el texto extraído (solo si fue procesado).",
    ),
):
    """Descarga o lee el texto de un contrato de ejemplo individual."""

    # ---- Validar extensión -----------------------------------------------
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Extensión '{ext or 'sin extensión'}' no soportada. Solo JPEG, PNG y PDF.",
        )

    # ---- Path traversal protection ----------------------------------------
    # Reconstruimos el path absoluto y verificamos que siga dentro del directorio
    safe_path = os.path.abspath(os.path.join(TEST_CONTRACTS_DIR, filename))
    if not safe_path.startswith(os.path.abspath(TEST_CONTRACTS_DIR) + os.sep):
        raise HTTPException(
            status_code=422,
            detail="Nombre de archivo inválido.",
        )

    # ---- Verificar que el archivo exista ----------------------------------
    if not os.path.isfile(safe_path):
        raise HTTPException(
            status_code=404,
            detail=f"El archivo '{filename}' no existe en los contratos de ejemplo.",
        )

    # ---- Modo raw: servir el archivo directamente -------------------------
    if mode == "raw":
        content_type = EXTENSION_CONTENT_TYPES.get(ext, "application/octet-stream")
        return FileResponse(
            path=safe_path,
            media_type=content_type,
            filename=filename,
        )

    # ---- Modo text: leer del caché en output/ ----------------------------
    stem            = os.path.splitext(filename)[0]          # "documento_1__original"
    cached_text_path = os.path.join(OUTPUT_DIR, f"{stem}_extracted.txt")

    if os.path.isfile(cached_text_path):
        with open(cached_text_path, encoding="utf-8") as f:
            text = f.read()
        return {
            "filename": filename,
            "source":   "cache",
            "text":     text,
        }

    # Sin texto cacheado: devolver info útil sin llamar a OpenAI
    return JSONResponse(
        status_code=200,
        content={
            "filename": filename,
            "source":   "not_available",
            "text":     None,
            "hint": (
                f"El texto de '{filename}' no ha sido extraído aún. "
                "Para generarlo, procesá el par correspondiente con "
                "POST /api/v1/analyze/sample (o POST /api/v1/analyze)."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Shared pipeline execution helper
# ---------------------------------------------------------------------------

def _execute_pipeline(
    original_path:   str,
    amendment_path:  str,
    naming_original: str,
    naming_amendment: str,
    save_files:      bool,
) -> ContractChangeOutput:
    """
    Ejecuta el pipeline de análisis y traduce todas las excepciones a HTTPException.
    Utilizado internamente por POST /api/v1/analyze y POST /api/v1/analyze/sample.
    """
    result = None
    try:
        logger.info("🔍 Iniciando pipeline...")
        result, text_original, text_amendment = run_pipeline(
            original_path=original_path,
            amendment_path=amendment_path,
        )
        logger.info("✅ Pipeline completado. Secciones modificadas: %s", result.sections_changed)

        if save_files:
            saved_paths = save_output_files(
                text_original=text_original,
                text_amendment=text_amendment,
                result=result,
                original_path=naming_original,
                amendment_path=naming_amendment,
                output_dir=OUTPUT_DIR,
            )
            logger.info(
                "💾 Archivos guardados: %s | %s | %s",
                os.path.basename(saved_paths["text_original"]),
                os.path.basename(saved_paths["text_amendment"]),
                os.path.basename(saved_paths["result"]),
            )

        return result

    except ImageParsingError as e:
        logger.error("❌ ImageParsingError: %s", e)
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "ImageParsingError",
                "message": str(e),
                "stage":   "image_parsing",
                "hint":    "Verificá que los archivos sean JPEG, PNG o PDF legibles y que contengan contratos.",
            },
        )

    except BadContractsError as e:
        logger.error("❌ BadContractsError: %s", e)
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "BadContractsError",
                "message": str(e),
                "stage":   "extraction_agent",
                "hint":    "Los contratos no son comparables: sus objetos son totalmente distintos.",
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
        logger.warning("⚠️  OutputSaveError (resultado disponible de todas formas): %s", e)
        return result  # pipeline fue exitoso, solo falló el guardado

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


# ---------------------------------------------------------------------------
# Endpoints de análisis
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/analyze",
    response_model=ContractChangeOutput,
    summary="Analizar contratos (upload)",
    description=(
        "Recibe los archivos del contrato original y la enmienda como `multipart/form-data`, "
        "ejecuta el pipeline completo y devuelve un JSON estructurado con los cambios detectados.\n\n"
        "**Formatos aceptados**: JPEG, PNG, PDF (incluyendo PDFs escaneados y multi-página)\n\n"
        "**Tiempo estimado**: 30-60 segundos (llamadas a OpenAI)\n\n"
        "Para usar contratos de ejemplo sin subir archivos, vé `POST /api/v1/analyze/sample`."
    ),
)
async def analyze_contracts(
    original:   UploadFile = File(..., description="Contrato original (JPEG, PNG o PDF)"),
    amendment:  UploadFile = File(..., description="Enmienda o adenda (JPEG, PNG o PDF)"),
    save_files: bool       = Query(
        default=True,
        description="Si es true, guarda los textos extraídos y el resultado JSON en output/.",
    ),
) -> ContractChangeOutput:
    """Analiza un par de contratos subidos por el usuario."""
    _validate_document_file(original,  "original")
    _validate_document_file(amendment, "amendment")

    logger.info(
        "📥 Upload — original: %s | amendment: %s | save_files: %s",
        original.filename, amendment.filename, save_files,
    )

    tmp_original_path  = None
    tmp_amendment_path = None
    try:
        tmp_original_path  = await _save_upload_to_tempfile(original)
        tmp_amendment_path = await _save_upload_to_tempfile(amendment)
        naming_original    = os.path.join(OUTPUT_DIR, original.filename  or "original.jpg")
        naming_amendment   = os.path.join(OUTPUT_DIR, amendment.filename or "amendment.jpg")
        return _execute_pipeline(
            original_path=tmp_original_path,
            amendment_path=tmp_amendment_path,
            naming_original=naming_original,
            naming_amendment=naming_amendment,
            save_files=save_files,
        )
    finally:
        for tmp_path in [tmp_original_path, tmp_amendment_path]:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


@app.post(
    "/api/v1/analyze/sample",
    response_model=ContractChangeOutput,
    summary="Analizar contratos de ejemplo",
    description=(
        "Ejecuta el pipeline usando un par de contratos de ejemplo disponibles en el servidor. "
        "No requiere subir archivos.\n\n"
        "Los pares disponibles se listan en `GET /api/v1/contracts`.\n\n"
        "**Tiempo estimado**: 30-60 segundos (llamadas a OpenAI)"
    ),
)
def analyze_sample_contracts(
    pair:       str  = Query(..., description="Nombre del par de ejemplo (ej: 'documento_1'). Ver GET /api/v1/contracts."),
    save_files: bool = Query(
        default=True,
        description="Si es true, guarda los textos extraídos y el resultado JSON en output/.",
    ),
) -> ContractChangeOutput:
    """Analiza un par de contratos de ejemplo pre-cargados en el servidor."""
    original_path, amendment_path = _find_sample_pair(pair)

    logger.info(
        "📥 Sample — pair: %s | save_files: %s",
        pair, save_files,
    )

    return _execute_pipeline(
        original_path=original_path,
        amendment_path=amendment_path,
        naming_original=original_path,
        naming_amendment=amendment_path,
        save_files=save_files,
    )

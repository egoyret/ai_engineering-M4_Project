"""
api.py
------
API REST del sistema de análisis de contratos.

Expone el pipeline como un endpoint HTTP usando FastAPI.
Ambas interfaces (CLI y API) comparten el mismo código a través de pipeline.py.

Endpoints:
    GET  /                      → Info del servicio
    GET  /health                → Health check
    GET  /api/v1/contracts      → Lista los contratos de ejemplo disponibles
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

from typing import Optional

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

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

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
            "contracts": "GET  /api/v1/contracts",
            "analyze":   "POST /api/v1/analyze",
            "health":    "GET  /health",
            "docs":      "GET  /docs",
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
        "de sus nombres. Estos pares pueden usarse directamente con el endpoint `/api/v1/analyze`."
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


@app.post(
    "/api/v1/analyze",
    response_model=ContractChangeOutput,
    summary="Analizar par de contratos",
    description=(
        "Ejecuta el pipeline completo y devuelve un JSON estructurado con los cambios detectados.\n\n"
        "**Dos modos de uso:**\n"
        "- **Upload**: envíá los campos `original` y `amendment` como archivos (multipart/form-data).\n"
        "- **Sample**: pasá el parámetro `?pair=documento_1` para usar un contrato de ejemplo "
        "(los disponibles se listan en `GET /api/v1/contracts`).\n\n"
        "Si se envían archivos Y `pair` al mismo tiempo, los archivos tienen prioridad.\n\n"
        "**Formatos aceptados**: JPEG, PNG, PDF (incluyendo PDFs escaneados y multi-página)\n\n"
        "**Tiempo estimado**: 30-60 segundos (llamadas a OpenAI)"
    ),
)
async def analyze_contracts(
    original:   Optional[UploadFile] = File(default=None, description="Contrato original (JPEG, PNG o PDF). Requerido si no se usa 'pair'."),
    amendment:  Optional[UploadFile] = File(default=None, description="Enmienda o adenda (JPEG, PNG o PDF). Requerido si no se usa 'pair'."),
    pair:       Optional[str]        = Query(
        default=None,
        description="Nombre de un par de ejemplo (ej: 'documento_1'). Ignorado si se suben archivos. Ver GET /api/v1/contracts para la lista.",
    ),
    save_files: bool                 = Query(
        default=True,
        description=(
            "Si es true (por defecto), guarda los textos extraídos y el resultado JSON "
            "en la carpeta output/. Si es false, sólo devuelve el JSON sin tocar disco."
        ),
    ),
) -> ContractChangeOutput:
    """
    Endpoint principal: ejecuta el pipeline en modo upload o modo sample.
    """
    # ------------------------------------------------------------------
    # Determinar modo y resolver paths
    # ------------------------------------------------------------------
    use_upload_mode = (original is not None or amendment is not None)
    tmp_original_path  = None
    tmp_amendment_path = None
    is_temp            = False          # True solo en modo upload
    naming_original    = None           # Path usado para nombrar los archivos de salida
    naming_amendment   = None

    if use_upload_mode:
        # Modo upload: ambos archivos son obligatorios
        if original is None or amendment is None:
            raise HTTPException(
                status_code=422,
                detail="En modo upload debés enviar tanto 'original' como 'amendment'. "
                       "Si querés usar un contrato de ejemplo, usá solo el parámetro 'pair'.",
            )
        _validate_document_file(original,  "original")
        _validate_document_file(amendment, "amendment")

        logger.info(
            "📥 Modo upload — original: %s | amendment: %s | save_files: %s",
            original.filename, amendment.filename, save_files,
        )

        tmp_original_path  = await _save_upload_to_tempfile(original)
        tmp_amendment_path = await _save_upload_to_tempfile(amendment)
        is_temp            = True
        naming_original    = os.path.join(OUTPUT_DIR, original.filename  or "original.jpg")
        naming_amendment   = os.path.join(OUTPUT_DIR, amendment.filename or "amendment.jpg")

    elif pair is not None:
        # Modo sample: resolvemos los paths desde data/test_contracts/
        tmp_original_path, tmp_amendment_path = _find_sample_pair(pair)
        is_temp          = False        # no son temp files, no borrar
        naming_original  = tmp_original_path
        naming_amendment = tmp_amendment_path

        logger.info(
            "📥 Modo sample — pair: %s | save_files: %s",
            pair, save_files,
        )

    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "Debés proveer alguna entrada: "
                "(a) los campos 'original' y 'amendment' como archivos, o "
                "(b) el parámetro '?pair=<nombre>' con un contrato de ejemplo. "
                "Consultá GET /api/v1/contracts para ver los pares disponibles."
            ),
        )

    # ------------------------------------------------------------------
    # Ejecutar el pipeline
    # ------------------------------------------------------------------
    try:
        logger.info("🔍 Iniciando pipeline...")

        result, text_original, text_amendment = run_pipeline(
            original_path=tmp_original_path,
            amendment_path=tmp_amendment_path,
        )

        logger.info("✅ Pipeline completado. Secciones modificadas: %s", result.sections_changed)

        # --- Guardar archivos en output/ (si save_files=True) ---
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

    # --- Traducir excepciones del pipeline a respuestas HTTP ---
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
    except BadContractsError as e:
        logger.error("❌ BadContractsError: %s", e)
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "BadContractsError",
                "message": str(e),
                "stage":   "extraction_agent",
                "hint":    "El contrato original no parece ser comparable con su enmienda ya que los objectos de los mismos son totalmente distintos.",
            },
        )

    except OutputSaveError as e:
        logger.warning("⚠ﺏ  OutputSaveError (resultado disponible de todas formas): %s", e)
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
        # Eliminar archivos temporales solo en modo upload
        if is_temp:
            for tmp_path in [tmp_original_path, tmp_amendment_path]:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

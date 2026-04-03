"""
pipeline.py
-----------
Módulo central del pipeline de análisis de contratos.

Contiene la lógica pura del pipeline, sin dependencias de CLI ni de HTTP.
Es importado tanto por main.py (CLI) como por api.py (FastAPI).

Funciones exportadas:
    run_pipeline(original_path, amendment_path)
        → (ContractChangeOutput, text_original, text_amendment)

    save_output_files(text_original, text_amendment, result,
                      original_path, amendment_path, output_dir)
        → dict con paths guardados
"""

import os
import json
import time

from dotenv import load_dotenv
from langfuse import Langfuse

from src.image_parser import parse_contract_image
from src.agents.contextualization_agent import run_contextualization_agent
from src.agents.extraction_agent import run_extraction_agent
from src.models import ContractChangeOutput
from src.exceptions import OutputSaveError

# ---------------------------------------------------------------------------
# Cargar variables de entorno (.env)
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers de output
# ---------------------------------------------------------------------------

def _get_output_paths(original_path: str, amendment_path: str, output_dir: str) -> dict:
    """
    Calcula los paths de salida a partir de los nombres de las imágenes.

    Estrategia de naming:
      - Prefijo común = prefijo compartido entre ambos stems de imagen.
        Ej: 'documento_1__original' y 'documento_1__enmienda'
             → prefijo común = 'documento_1'
      - Archivos:
          <output_dir>/<stem_original>_extracted.txt
          <output_dir>/<stem_enmienda>_extracted.txt
          <output_dir>/<prefijo_comun>_result.json
    """
    stem_original  = os.path.splitext(os.path.basename(original_path))[0]
    stem_amendment = os.path.splitext(os.path.basename(amendment_path))[0]

    common_prefix = os.path.commonprefix([stem_original, stem_amendment]).rstrip("_-")
    if not common_prefix:
        common_prefix = stem_original

    return {
        "text_original":  os.path.join(output_dir, f"{stem_original}_extracted.txt"),
        "text_amendment": os.path.join(output_dir, f"{stem_amendment}_extracted.txt"),
        "result":         os.path.join(output_dir, f"{common_prefix}_result.json"),
    }


def save_output_files(
    text_original: str,
    text_amendment: str,
    result: ContractChangeOutput,
    original_path: str,
    amendment_path: str,
    output_dir: str,
) -> dict:
    """
    Guarda en disco los tres archivos de salida del pipeline.

    Los archivos se sobreescriben si ya existen (idempotente).

    Returns:
        Dict con claves 'text_original', 'text_amendment', 'result'
        apuntando a los paths absolutos de los archivos guardados.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        paths = _get_output_paths(original_path, amendment_path, output_dir)

        with open(paths["text_original"], "w", encoding="utf-8") as f:
            f.write(f"# Texto extraído de: {os.path.basename(original_path)}\n")
            f.write("=" * 60 + "\n\n")
            f.write(text_original)

        with open(paths["text_amendment"], "w", encoding="utf-8") as f:
            f.write(f"# Texto extraído de: {os.path.basename(amendment_path)}\n")
            f.write("=" * 60 + "\n\n")
            f.write(text_amendment)

        with open(paths["result"], "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

        return paths

    except OSError as e:
        raise OutputSaveError(
            f"❌ No se pudieron guardar los archivos de salida en '{output_dir}'.\n"
            f"   Verificá los permisos del directorio.\n"
            f"   Detalle técnico: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(original_path: str, amendment_path: str) -> tuple:
    """
    Ejecuta el pipeline completo de análisis de contratos.

    Crea una traza raíz en Langfuse y ejecuta secuencialmente:
      1. Parsing multimodal del contrato original  (GPT-4o Vision)
      2. Parsing multimodal de la enmienda         (GPT-4o Vision)
      3. Agente 1: Contextualización               (gpt-4o-mini, LCEL)
      4. Agente 2: Extracción de cambios           (gpt-4o-mini, LCEL + Pydantic)

    Args:
        original_path:  Ruta al archivo de imagen del contrato original.
        amendment_path: Ruta al archivo de imagen de la enmienda.

    Returns:
        Tupla (result, text_original, text_amendment):
          - result:         ContractChangeOutput validado por Pydantic.
          - text_original:  Texto extraído del contrato original.
          - text_amendment: Texto extraído de la enmienda.

    Toda la ejecución queda registrada en Langfuse bajo una traza raíz
    "contract-analysis" con spans hijos para cada etapa (API Langfuse v4).

    Estructura de trazas en Langfuse:
    contract-analysis  (trace raíz)
    ├── parse_original_contract   (span)
    ├── parse_amendment_contract  (span)
    ├── contextualization_agent   (span)
    └── extraction_agent          (span)

    """
    langfuse = Langfuse()
    pipeline_start = time.time()

    with langfuse.start_as_current_observation(
        name="contract-analysis",
        as_type="span",
        input={
            "original_image":  os.path.basename(original_path),
            "amendment_image": os.path.basename(amendment_path),
        },
    ):
        # --- Etapa 1: Parsing original ---
        text_original, tokens_parse_original = parse_contract_image(
            image_path=original_path,
            langfuse=langfuse,
            span_name="parse_original_contract",
        )

        # --- Etapa 2: Parsing enmienda ---
        text_amendment, tokens_parse_amendment = parse_contract_image(
            image_path=amendment_path,
            langfuse=langfuse,
            span_name="parse_amendment_contract",
        )

        # --- Etapa 3: Agente 1 — Contextualización ---
        context_map, tokens_contextualization = run_contextualization_agent(
            text_original=text_original,
            text_amendment=text_amendment,
            langfuse=langfuse,
        )

        # --- Etapa 4: Agente 2 — Extracción ---
        result, tokens_extraction = run_extraction_agent(
            context_map=context_map,
            text_original=text_original,
            text_amendment=text_amendment,
            langfuse=langfuse,
        )

        # --- Acumular tokens y cerrar traza raíz ---
        def _safe_get(d: dict, key: str) -> int:
            return d.get(key) or 0

        all_usages = [
            tokens_parse_original,
            tokens_parse_amendment,
            tokens_contextualization,
            tokens_extraction,
        ]
        grand_total = {
            "input_tokens":  sum(_safe_get(t, "input_tokens")  for t in all_usages),
            "output_tokens": sum(_safe_get(t, "output_tokens") for t in all_usages),
            "total_tokens":  sum(_safe_get(t, "total_tokens")  for t in all_usages),
        }

        langfuse.update_current_span(
            output={
                "sections_changed":      result.sections_changed,
                "topics_touched":        result.topics_touched,
                "summary_of_the_change": result.summary_of_the_change,
            },
            metadata={
                "total_latency_ms": int((time.time() - pipeline_start) * 1000),
                "status": "success",
                "token_usage": {
                    "parse_original_contract":  tokens_parse_original,
                    "parse_amendment_contract": tokens_parse_amendment,
                    "contextualization_agent":  tokens_contextualization,
                    "extraction_agent":         tokens_extraction,
                    "grand_total":              grand_total,
                },
            },
        )

    langfuse.flush()
    return result, text_original, text_amendment

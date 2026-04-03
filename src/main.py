"""
main.py
-------
Punto de entrada del pipeline de análisis de contratos.

Orquesta el flujo completo en 4 etapas principales:
  1. Parsing multimodal del contrato original     → GPT-4o Vision
  2. Parsing multimodal de la enmienda            → GPT-4o Vision
  3. Agente 1: Contextualización                  → gpt-4o-mini (LCEL)
  4. Agente 2: Extracción de cambios              → gpt-4o-mini (LCEL, structured output)

Toda la ejecución queda registrada en Langfuse bajo una traza raíz
"contract-analysis" con spans hijos para cada etapa (API Langfuse v4).

Estructura de trazas en Langfuse:
  contract-analysis  (trace raíz)
  ├── parse_original_contract   (span)
  ├── parse_amendment_contract  (span)
  ├── contextualization_agent   (span)
  └── extraction_agent          (span)

Archivos de salida generados en output/:
  <prefijo>__original_extracted.txt  → texto extraído del contrato original
  <prefijo>__enmienda_extracted.txt  → texto extraído de la enmienda
  <prefijo>__result.json             → resultado final validado por Pydantic

Uso:
    # Procesa el par documento_1 (por defecto)
    python src/main.py

    # Procesa un par de imágenes específico
    python src/main.py ruta/original.jpg ruta/enmienda.jpg
"""

import os
import sys
import json
import time

from dotenv import load_dotenv
from langfuse import Langfuse

# Asegurar que la raíz del proyecto está en el path cuando se ejecuta
# desde la raíz con `python src/main.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_parser import parse_contract_image
from src.agents.contextualization_agent import run_contextualization_agent
from src.agents.extraction_agent import run_extraction_agent
from src.models import ContractChangeOutput
from src.exceptions import (
    ContractPipelineError,
    ImageParsingError,
    ContextualizationError,
    ExtractionError,
    OutputSaveError,
)

# ---------------------------------------------------------------------------
# Cargar variables de entorno (.env)
# Lee: OPENAI_API_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
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
             → prefijo común = 'documento_1__'
      - Archivos:
          <output_dir>/<stem_original>_extracted.txt
          <output_dir>/<stem_enmienda>_extracted.txt
          <output_dir>/<prefijo_comun>result.json

    Args:
        original_path:  Ruta a la imagen del contrato original.
        amendment_path: Ruta a la imagen de la enmienda.
        output_dir:     Directorio donde guardar los archivos.

    Returns:
        Dict con claves 'text_original', 'text_amendment', 'result'.
    """
    stem_original  = os.path.splitext(os.path.basename(original_path))[0]
    stem_amendment = os.path.splitext(os.path.basename(amendment_path))[0]

    # Prefijo compartido entre los dos stems (ej. "documento_1__")
    common_prefix = os.path.commonprefix([stem_original, stem_amendment])
    # Limpiar guiones/underscores finales del prefijo
    common_prefix = common_prefix.rstrip("_-")

    # Si no hay prefijo común, usar el stem del original
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

    Los archivos se sobreescriben si ya existen (misma corrida con mismas imágenes).

    Args:
        text_original:   Texto extraído del contrato original.
        text_amendment:  Texto extraído de la enmienda.
        result:          Objeto ContractChangeOutput con los cambios detectados.
        original_path:   Ruta a la imagen original (para derivar el nombre).
        amendment_path:  Ruta a la imagen de la enmienda (para derivar el nombre).
        output_dir:      Directorio donde guardar los archivos.

    Returns:
        Dict con los paths de los archivos guardados.
    """
    # Crear el directorio output/ si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Calcular paths de salida
    paths = _get_output_paths(original_path, amendment_path, output_dir)

    # 1. Guardar texto extraído del contrato original
    with open(paths["text_original"], "w", encoding="utf-8") as f:
        f.write(f"# Texto extraído de: {os.path.basename(original_path)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text_original)

    # 2. Guardar texto extraído de la enmienda
    with open(paths["text_amendment"], "w", encoding="utf-8") as f:
        f.write(f"# Texto extraído de: {os.path.basename(amendment_path)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text_amendment)

    # 3. Guardar resultado final en JSON
    with open(paths["result"], "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    return paths


def run_pipeline(original_path: str, amendment_path: str) -> tuple:
    """
    Ejecuta el pipeline completo de análisis de contratos.

    Crea una traza raíz en Langfuse y ejecuta secuencialmente cada etapa
    del pipeline, registrando spans hijos con input, output y metadata.

    Args:
        original_path:  Ruta a la imagen del contrato original.
        amendment_path: Ruta a la imagen de la enmienda/adenda.

    Returns:
        Tupla (ContractChangeOutput, text_original, text_amendment):
          - result:          Objeto Pydantic con los cambios detectados.
          - text_original:   Texto extraído del contrato original.
          - text_amendment:  Texto extraído de la enmienda.
    """
    print("\n" + "=" * 60)
    print("  SISTEMA DE ANÁLISIS DE CONTRATOS — Pipeline Iniciado")
    print("=" * 60)
    print(f"  📄 Contrato original : {os.path.basename(original_path)}")
    print(f"  📝 Enmienda          : {os.path.basename(amendment_path)}")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # Inicializar Langfuse v4
    # Toma automáticamente del entorno:
    #   LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
    # -----------------------------------------------------------------------
    langfuse = Langfuse()

    pipeline_start = time.time()
    result = None

    # -----------------------------------------------------------------------
    # Traza raíz: engloba todo el pipeline bajo "contract-analysis"
    # Todas las llamadas a start_as_current_observation() dentro de este
    # bloque crearán spans hijos automáticamente (vía contexto OpenTelemetry).
    # -----------------------------------------------------------------------
    # Crear la traza raíz "contract-analysis".
    # En Langfuse v4 la traza se genera automáticamente al iniciar la primera
    # observación. as_type="span" es el tipo raíz que actúa como trace en el
    # dashboard. set_current_trace_io() fija los I/O del nodo raíz.
    with langfuse.start_as_current_observation(
        name="contract-analysis",
        as_type="span",
        input={
            "original_image":  os.path.basename(original_path),
            "amendment_image": os.path.basename(amendment_path),
        },
    ):

        # -------------------------------------------------------------------
        # ETAPA 1: Parsing multimodal — Contrato original
        # -------------------------------------------------------------------
        print("🔍 [Etapa 1/4] Parseando contrato original con GPT-4o Vision...")
        text_original, tokens_parse_original = parse_contract_image(
            image_path=original_path,
            langfuse=langfuse,
            span_name="parse_original_contract",
        )
        print(f"   ✅ Texto extraído ({len(text_original)} caracteres)\n")

        # -------------------------------------------------------------------
        # ETAPA 2: Parsing multimodal — Enmienda
        # -------------------------------------------------------------------
        print("🔍 [Etapa 2/4] Parseando enmienda con GPT-4o Vision...")
        text_amendment, tokens_parse_amendment = parse_contract_image(
            image_path=amendment_path,
            langfuse=langfuse,
            span_name="parse_amendment_contract",
        )
        print(f"   ✅ Texto extraído ({len(text_amendment)} caracteres)\n")

        # -------------------------------------------------------------------
        # ETAPA 3: Agente 1 — Contextualización
        # -------------------------------------------------------------------
        print("🤖 [Etapa 3/4] Agente 1: construyendo mapa contextual...")
        context_map, tokens_contextualization = run_contextualization_agent(
            text_original=text_original,
            text_amendment=text_amendment,
            langfuse=langfuse,
        )
        print(f"   ✅ Mapa contextual generado ({len(context_map)} caracteres)\n")

        # -------------------------------------------------------------------
        # ETAPA 4: Agente 2 — Extracción de cambios
        # -------------------------------------------------------------------
        print("🤖 [Etapa 4/4] Agente 2: extrayendo y estructurando cambios...")
        result, tokens_extraction = run_extraction_agent(
            context_map=context_map,
            text_original=text_original,
            text_amendment=text_amendment,
            langfuse=langfuse,
        )
        print("   ✅ Cambios extraídos y validados con Pydantic\n")

        # -------------------------------------------------------------------
        # Acumular token usage de todas las etapas y calcular el grand total
        # -------------------------------------------------------------------
        def _safe_get(d: dict, key: str) -> int:
            """Retorna el valor de la clave o 0 si no existe / es None."""
            return d.get(key) or 0

        all_token_usages = [
            tokens_parse_original,
            tokens_parse_amendment,
            tokens_contextualization,
            tokens_extraction,
        ]

        grand_total_tokens = {
            "input_tokens":  sum(_safe_get(t, "input_tokens")  for t in all_token_usages),
            "output_tokens": sum(_safe_get(t, "output_tokens") for t in all_token_usages),
            "total_tokens":  sum(_safe_get(t, "total_tokens")  for t in all_token_usages),
        }

        # Registrar output final, latencia y token usage total en la traza raíz
        total_latency_ms = int((time.time() - pipeline_start) * 1000)
        langfuse.update_current_span(
            output={
                "sections_changed":      result.sections_changed,
                "topics_touched":        result.topics_touched,
                "summary_of_the_change": result.summary_of_the_change,
            },
            metadata={
                "total_latency_ms": total_latency_ms,
                "status":           "success",
                # Desglose por etapa y total acumulado de todo el pipeline
                "token_usage": {
                    "parse_original_contract":  tokens_parse_original,
                    "parse_amendment_contract": tokens_parse_amendment,
                    "contextualization_agent":  tokens_contextualization,
                    "extraction_agent":         tokens_extraction,
                    "grand_total":              grand_total_tokens,
                },
            },
        )

    # Asegurar que Langfuse envía todos los eventos pendientes antes de salir
    langfuse.flush()

    return result, text_original, text_amendment


def main():
    """
    Función principal. Determina las rutas de las imágenes y ejecuta el pipeline.

    Si se pasan dos argumentos desde la línea de comandos, los usa como rutas.
    Si no, usa el par documento_1 de la carpeta data/test_contracts por defecto.
    """
    # Directorio raíz del proyecto (un nivel arriba de /src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(project_root, "data", "test_contracts")

    # Resolver rutas: argumentos CLI o valores por defecto
    if len(sys.argv) == 3:
        original_path  = sys.argv[1]
        amendment_path = sys.argv[2]
    else:
        original_path  = os.path.join(default_data_dir, "documento_1__original.jpg")
        amendment_path = os.path.join(default_data_dir, "documento_1__enmienda.jpg")

    # Validar que los archivos existen antes de iniciar
    for path in [original_path, amendment_path]:
        if not os.path.exists(path):
            print(f"\n❌ Error: no se encontró el archivo de imagen: {path}")
            print("   Verificá la ruta e intentá nuevamente.")
            sys.exit(1)

    # Directorio de salida: output/ en la raíz del proyecto
    output_dir = os.path.join(project_root, "output")

    # -------------------------------------------------------------------
    # Ejecutar el pipeline con manejo de excepciones por tipo
    # -------------------------------------------------------------------
    try:
        result, text_original, text_amendment = run_pipeline(original_path, amendment_path)

    except ImageParsingError as e:
        print(f"\n{'='*60}")
        print("  ⚠️  ERROR EN PARSING DE IMAGEN")
        print(f"{'='*60}")
        print(str(e))
        print("\n💡 Sugerencias:")
        print("   • Verificá que la imagen no esté corrupta")
        print("   • Aseguráte de que OPENAI_API_KEY es válida en el archivo .env")
        print("   • Confirmá que la imagen sea un contrato legible (JPEG/PNG)")
        sys.exit(1)

    except ContextualizationError as e:
        print(f"\n{'='*60}")
        print("  ⚠️  ERROR EN AGENTE 1 (CONTEXTUALIZACIÓN)")
        print(f"{'='*60}")
        print(str(e))
        print("\n💡 Sugerencias:")
        print("   • Verificá que OPENAI_API_KEY tiene cuota disponible")
        print("   • Intentá volver a ejecutar el pipeline")
        sys.exit(1)

    except ExtractionError as e:
        print(f"\n{'='*60}")
        print("  ⚠️  ERROR EN AGENTE 2 (EXTRACCIÓN)")
        print(f"{'='*60}")
        print(str(e))
        print("\n💡 Sugerencias:")
        print("   • El modelo no pudo producir el JSON requerido con estos documentos")
        print("   • Intentá volver a ejecutar el pipeline")
        print("   • Si el problema persiste, revisá el dashboard de Langfuse para ver el output crudo")
        sys.exit(1)

    except ContractPipelineError as e:
        # Catch genérico para cualquier otra excepción del pipeline
        print(f"\n{'='*60}")
        print("  ⚠️  ERROR EN EL PIPELINE")
        print(f"{'='*60}")
        print(str(e))
        sys.exit(1)

    except Exception as e:
        # Error completamente inesperado (bug, problema de entorno, etc.)
        print(f"\n{'='*60}")
        print("  🔴 ERROR INESPERADO")
        print(f"{'='*60}")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Mensaje: {e}")
        print("\n💡 Este es un error no esperado. Revisá el stack trace para más detalles.")
        raise  # Re-raise para mostrar traceback completo

    # -------------------------------------------------------------------
    # Pipeline exitoso: guardar archivos y mostrar resultados
    # -------------------------------------------------------------------
    try:
        saved_paths = save_output_files(
            text_original=text_original,
            text_amendment=text_amendment,
            result=result,
            original_path=original_path,
            amendment_path=amendment_path,
            output_dir=output_dir,
        )
    except OutputSaveError as e:
        print(str(e))
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Output final: imprimir el JSON validado por Pydantic
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  📋 RESULTADO FINAL (validado por Pydantic)")
    print("=" * 60)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print("=" * 60)

    # Mostrar rutas de archivos guardados
    print("\n💾 Archivos guardados en output/:")
    print(f"   📄 {os.path.basename(saved_paths['text_original'])}")
    print(f"   📄 {os.path.basename(saved_paths['text_amendment'])}")
    print(f"   📋 {os.path.basename(saved_paths['result'])}")

    print("\n✅ Pipeline completado exitosamente.")
    print("   Revisá el dashboard de Langfuse para ver las trazas completas:")
    print("   → https://us.cloud.langfuse.com\n")


if __name__ == "__main__":
    main()


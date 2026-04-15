"""
main.py
-------
Punto de entrada CLI del sistema de análisis de contratos.

Orquesta el flujo completo llamando a pipeline.py, que contiene la lógica
central compartida también con api.py (FastAPI).

Uso:
    # Procesa el par documento_1 (por defecto)
    python src/main.py

    # Procesa un par de imágenes específico
    python src/main.py ruta/original.jpg ruta/enmienda.jpg
"""

import os
import sys
import json

# Asegurar que la raíz del proyecto está en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline, save_output_files
from src.exceptions import (
    ContractPipelineError,
    ImageParsingError,
    ContextualizationError,
    ExtractionError,
    BadContractsError,
    OutputSaveError,
)


def main():
    """
    Función principal del CLI. Determina las rutas de las imágenes,
    ejecuta el pipeline y guarda los resultados en output/.

    Si se pasan dos argumentos desde la línea de comandos, los usa como rutas.
    Si no, usa el par documento_1 de la carpeta data/test_contracts por defecto.

    Formatos aceptados: JPEG, PNG, PDF (incluyendo PDFs escaneados y multi-página).
    Se puede mezclar: ej. original como PDF y enmienda como JPEG.
    """
    project_root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(project_root, "data", "test_contracts")
    output_dir       = os.path.join(project_root, "output")

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
            print(f"\n❌ Error: no se encontró el archivo: {path}")
            print("   Verificá la ruta e intentá nuevamente.")
            print("   Formatos aceptados: JPEG, PNG, PDF.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  SISTEMA DE ANÁLISIS DE CONTRATOS — Pipeline Iniciado")
    print("=" * 60)
    print(f"  📄 Contrato original : {os.path.basename(original_path)}")
    print(f"  📝 Enmienda          : {os.path.basename(amendment_path)}")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------
    # Ejecutar el pipeline con manejo de excepciones por tipo
    # -------------------------------------------------------------------
    try:
        result, text_original, text_amendment = run_pipeline(original_path, amendment_path, source="cli")
        print("   ✅ Pipeline completado.\n")

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
        print("   • Si el problema persiste, revisá el dashboard de Langfuse")
        sys.exit(1)
    
    except BadContractsError as e:
        print(f"\n{'='*60}")
        print("  ⚠️  ERROR EN AGENTE 2 (EXTRACCIÓN)")
        print(f"{'='*60}")
        print(str(e))
        print("\n💡 Sugerencias:")
        print("   • Los contratos ingresados no se corresponden")
        print("   • El contrato original no parece ser comparable con su enmienda ya que los objectos de los mismos son totalmente distintos.")
        sys.exit(1)

    except ContractPipelineError as e:
        print(f"\n{'='*60}")
        print("  ⚠️  ERROR EN EL PIPELINE")
        print(f"{'='*60}")
        print(str(e))
        sys.exit(1)

    except Exception as e:
        print(f"\n{'='*60}")
        print("  🔴 ERROR INESPERADO")
        print(f"{'='*60}")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Mensaje: {e}")
        print("\n💡 Este es un error no esperado. Revisá el stack trace para más detalles.")
        raise

    # -------------------------------------------------------------------
    # Guardar archivos en output/
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

    # -------------------------------------------------------------------
    # Output final
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  📋 RESULTADO FINAL (validado por Pydantic)")
    print("=" * 60)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print("=" * 60)

    print("\n💾 Archivos guardados en output/:")
    print(f"   📄 {os.path.basename(saved_paths['text_original'])}")
    print(f"   📄 {os.path.basename(saved_paths['text_amendment'])}")
    print(f"   📋 {os.path.basename(saved_paths['result'])}")

    print("\n✅ Pipeline completado exitosamente.")
    print("   Revisá el dashboard de Langfuse para ver las trazas completas:")
    print("   → https://us.cloud.langfuse.com\n")


if __name__ == "__main__":
    main()

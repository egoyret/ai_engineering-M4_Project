"""
image_parser.py
---------------
Módulo responsable del parsing multimodal de documentos de contratos.

Soporta los siguientes formatos de entrada:
  - Imágenes: JPEG, PNG
  - Documentos: PDF (incluyendo PDFs escaneados, procesados por Vision)

Utiliza GPT-4o a través de la API de OpenAI Responses:
  - Imágenes → content type "input_image" (base64 inline)
  - PDFs     → content type "input_file"  (base64 inline, procesado nativamente)

GPT-4o maneja PDFs multi-página de forma automática en una sola llamada.

Cada llamada queda registrada como un span hijo dentro de la traza
activa de Langfuse usando la API de Langfuse v4 (basada en OpenTelemetry).
"""

import os
import base64
import time

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from langfuse import Langfuse

from src.exceptions import ImageParsingError

# ---------------------------------------------------------------------------
# Cargar variables de entorno (.env) antes de instanciar el cliente.
# Esto garantiza que OPENAI_API_KEY esté disponible aunque este módulo
# sea importado antes de que main.py llame a load_dotenv().
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Configuración del cliente OpenAI
# ---------------------------------------------------------------------------
client = OpenAI()

# Modelo con capacidad Vision que se usará para el parsing de imágenes
VISION_MODEL = "gpt-4o"


def encode_file_to_base64(file_path: str) -> str:
    """
    Codifica un archivo binario (imagen o PDF) a Base64.

    Args:
        file_path: Ruta al archivo (JPEG, PNG o PDF).

    Returns:
        String Base64 del contenido del archivo.
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_contract_image(
    image_path: str,
    langfuse: Langfuse,
    span_name: str,
) -> tuple:  # (extracted_text: str, token_usage: dict)
    """
    Procesa un documento de contrato con GPT-4o y extrae su texto completo.

    Soporta imágenes (JPEG/PNG) y PDFs (incluidos PDFs escaneados).
    Para imágenes usa el content type 'input_image'; para PDFs usa 'input_file',
    que GPT-4o procesa nativamente incluyendo documentos multi-página.

    Registra la ejecución como un span de Langfuse (API v4 / OpenTelemetry)
    con: input, output (preview), latencia, tipo de archivo y tokens.

    Args:
        image_path:  Ruta al archivo del contrato (JPEG, PNG o PDF).
        langfuse:    Instancia de Langfuse para crear observaciones.
        span_name:   Nombre descriptivo del span (ej. "parse_original_contract").

    Returns:
        Tupla (extracted_text, token_usage):
          - extracted_text: texto completo del contrato como string.
          - token_usage:    dict con input_tokens, output_tokens, total_tokens.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {image_path}")

    # Determinar tipo de archivo por extensión
    ext = os.path.splitext(image_path)[1].lower()
    is_pdf = ext == ".pdf"

    # Codificar el archivo a Base64 (funciona igual para imágenes y PDFs)
    base64_data = encode_file_to_base64(image_path)

    # Construir el content block adecuado según el tipo de archivo
    if is_pdf:
        file_content_block = {
            "type": "input_file",
            "filename": os.path.basename(image_path),
            "file_data": f"data:application/pdf;base64,{base64_data}",
        }
    else:
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        file_content_block = {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{base64_data}",
        }

    start_time = time.time()

    # ------------------------------------------------------------------
    # Llamada a la API de OpenAI con capacidad Vision (Responses API)
    # ------------------------------------------------------------------
    with langfuse.start_as_current_observation(
        as_type="span",
        name=span_name,
        input={
            "file_path": image_path,
            "file_type": "pdf" if is_pdf else "image",
            "model":     VISION_MODEL,
        },
    ):
        try:
            response = client.responses.create(
                model=VISION_MODEL,
                instructions=(
                    "Eres un asistente especializado en análisis de documentos legales. "
                    "Tu tarea es extraer el texto completo del contrato que aparece en el documento "
                    "(puede ser una imagen o un PDF, posiblemente con múltiples páginas), "
                    "de la manera más fiel y precisa posible, conservando la estructura "
                    "(títulos, numeración, cláusulas, párrafos). "
                    "No hagas resúmenes ni interpretaciones. "
                    "Devuelve únicamente el texto extraído, en el mismo idioma del documento."
                ),
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Extrae el texto completo de este documento de contrato "
                                    "(imagen o PDF), preservando su estructura original."
                                ),
                            },
                            file_content_block,
                        ],
                    }
                ],
            )

        except OpenAIError as e:
            # Error de la API de OpenAI (autenticación, rate limit, timeout, etc.)
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"OpenAI API error en '{span_name}': {e}",
            )
            raise ImageParsingError(
                f"❌ Error al llamar a la API de OpenAI para '{os.path.basename(image_path)}': {e}"
            ) from e

        except Exception as e:
            # Cualquier otro error inesperado (codificación, I/O, etc.)
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"Error inesperado en '{span_name}': {e}",
            )
            raise ImageParsingError(
                f"❌ Error inesperado durante el parsing de '{os.path.basename(image_path)}': {e}"
            ) from e

        latency_ms = int((time.time() - start_time) * 1000)
        extracted_text = response.output_text

        # Validar que el modelo devolvió texto (no una respuesta vacía)
        if not extracted_text or not extracted_text.strip() or extracted_text == "Lo siento, no puedo procesar esta solicitud.":
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"Respuesta vacía del modelo en '{span_name}'",
            )
            file_type_hint = "PDF legible" if is_pdf else "imagen legible"
            raise ImageParsingError(
                f"❌ GPT-4o no pudo extraer texto de '{os.path.basename(image_path)}'. "
                f"Verificá que el archivo sea un {file_type_hint} y contenga un contrato."
            )

        # Extraer uso de tokens si está disponible
        token_usage = {}
        if hasattr(response, "usage") and response.usage:
            token_usage = {
                "input_tokens":  getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
                "total_tokens":  getattr(response.usage, "total_tokens", None),
            }

        # Actualizar el span con el output una vez extraído el texto
        langfuse.update_current_span(
            output={"extracted_text_preview": extracted_text[:500] + "..."},
            metadata={
                "model":             VISION_MODEL,
                "file_type":         "pdf" if is_pdf else "image",
                "latency_ms":        latency_ms,
                "token_usage":       token_usage,
                "file_size_bytes":   os.path.getsize(image_path),
            },
        )

    return extracted_text, token_usage
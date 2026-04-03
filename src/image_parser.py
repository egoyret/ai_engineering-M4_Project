"""
image_parser.py
---------------
Módulo responsable del parsing multimodal de imágenes de contratos.

Utiliza GPT-4o (Vision) a través de la API de OpenAI Responses
para extraer el texto completo de cada imagen de contrato.

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


def encode_image_to_base64(image_path: str) -> str:
    """
    Codifica una imagen a Base64 para enviarla inline a la API de OpenAI.

    Args:
        image_path: Ruta absoluta o relativa al archivo de imagen (JPEG/PNG).

    Returns:
        String Base64 de la imagen.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_contract_image(
    image_path: str,
    langfuse: Langfuse,
    span_name: str,
) -> tuple:  # (extracted_text: str, token_usage: dict)
    """
    Procesa una imagen de contrato con GPT-4o Vision y extrae su texto completo.

    Registra la ejecución como un span de Langfuse (API v4 / OpenTelemetry)
    con: input, output (preview), latencia y tokens.

    Args:
        image_path:  Ruta al archivo de imagen del contrato.
        langfuse:    Instancia de Langfuse para crear observaciones.
        span_name:   Nombre descriptivo del span (ej. "parse_original_contract").

    Returns:
        Tupla (extracted_text, token_usage):
          - extracted_text: texto completo del contrato como string.
          - token_usage:    dict con input_tokens, output_tokens, total_tokens.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")

    # Determinar tipo MIME según extensión
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    # Codificar la imagen a Base64
    base64_image = encode_image_to_base64(image_path)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Llamada a la API de OpenAI con capacidad Vision (Responses API)
    # ------------------------------------------------------------------
    with langfuse.start_as_current_observation(
        as_type="span",
        name=span_name,
        input={"image_path": image_path, "model": VISION_MODEL},
    ):
        try:
            response = client.responses.create(
                model=VISION_MODEL,
                instructions=(
                    "Eres un asistente especializado en análisis de documentos legales. "
                    "Tu tarea es extraer el texto completo del contrato que aparece en la imagen, "
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
                                    "Extrae el texto completo de esta imagen de contrato, "
                                    "preservando su estructura original."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{base64_image}",
                            },
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
                f"❌ Error al llamar a la API de OpenAI Vision para '{os.path.basename(image_path)}': {e}"
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
            raise ImageParsingError(
                f"❌ GPT-4o no pudo extraer texto de '{os.path.basename(image_path)}'. "
                "Verificá que la imagen sea legible y contenga un contrato."
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
                "model":            VISION_MODEL,
                "latency_ms":       latency_ms,
                "token_usage":      token_usage,
                "image_size_bytes": os.path.getsize(image_path),
            },
        )

    return extracted_text, token_usage
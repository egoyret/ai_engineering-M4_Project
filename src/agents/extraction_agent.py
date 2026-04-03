"""
agents/extraction_agent.py
---------------------------
Agente 2: Extracción de cambios

Responsabilidad: usando el mapa contextual producido por el Agente 1
junto con los textos completos de ambos documentos, identificar y describir
con precisión cada cambio introducido por la enmienda:
  - Adiciones (texto nuevo que no estaba en el original)
  - Eliminaciones (texto del original que fue removido)
  - Modificaciones (texto que fue alterado)

El output es un objeto ContractChangeOutput validado con Pydantic,
producido mediante structured output de LangChain (with_structured_output).

Token usage: se usa include_raw=True en with_structured_output para recibir
tanto el objeto Pydantic parseado como el AIMessage original con usage_metadata.

La observabilidad usa Langfuse v4 (API basada en OpenTelemetry).
"""

import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langfuse import Langfuse
from pydantic import ValidationError

from src.models import ContractChangeOutput
from src.exceptions import ExtractionError

# Modelo de texto para este agente
AGENT_MODEL = "gpt-4o-mini"


def run_extraction_agent(
    context_map: str,
    text_original: str,
    text_amendment: str,
    langfuse: Langfuse,
) -> tuple:  # (result: ContractChangeOutput, token_usage: dict)
    """
    Ejecuta el Agente 2: extrae y estructura todos los cambios de la enmienda.

    Usa structured output (with_structured_output) de LangChain para que el LLM
    devuelva directamente un objeto Pydantic ContractChangeOutput validado.

    Crea un span hijo en la traza activa de Langfuse (v4) con input/output/latencia.

    Args:
        context_map:     Mapa contextual producido por el Agente 1.
        text_original:   Texto extraído del contrato original.
        text_amendment:  Texto extraído de la enmienda.
        langfuse:        Instancia de Langfuse para registrar observaciones.

    Returns:
        Tupla (result, token_usage):
          - result:      objeto ContractChangeOutput validado con Pydantic.
          - token_usage: dict con input_tokens, output_tokens, total_tokens.
    """
    # ------------------------------------------------------------------
    # Definir el prompt del Agente 2
    # ------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Eres un abogado especializado en análisis de contratos y enmiendas legales.
Tu tarea es identificar, aislar y describir con precisión cada cambio introducido
por la enmienda respecto al contrato original.

Recibirás:
1. Un mapa contextual que describe la estructura de ambos documentos (producido por un agente previo).
2. El texto completo del contrato original.
3. El texto completo de la enmienda.

Debes producir un análisis estructurado que incluya:
- sections_changed: lista de las secciones/cláusulas que fueron modificadas (usa los nombres/números del documento).
- topics_touched: lista de las categorías legales o comerciales afectadas (ej: "plazos", "precio", "penalidades", "partes", "vigencia").
- summary_of_the_change: descripción detallada y precisa de TODOS los cambios. Distingue claramente entre:
    * ADICIÓN: contenido nuevo que no existía en el original.
    * ELIMINACIÓN: contenido del original que fue removido.
    * MODIFICACIÓN: contenido que fue alterado (indica el valor anterior y el nuevo).

Sé específico y exhaustivo. Responde en el mismo idioma de los documentos."""
        ),
        (
            "human",
            """=== MAPA CONTEXTUAL (análisis de estructura) ===
{context_map}

=== CONTRATO ORIGINAL ===
{text_original}

=== ENMIENDA / ADENDA ===
{text_amendment}

Analiza los cambios y produce el output estructurado requerido."""
        ),
    ])

    # ------------------------------------------------------------------
    # Construir la chain con LCEL + structured output con include_raw=True
    #
    # with_structured_output(ContractChangeOutput) instruye al LLM a
    # devolver JSON compatible con el schema Pydantic. LangChain lo
    # deserializa y valida automáticamente, retornando el objeto Pydantic.
    #
    # include_raw=True hace que la chain devuelva un dict con:
    #   - 'parsed':        objeto ContractChangeOutput validado por Pydantic
    #   - 'raw':           AIMessage original (contiene usage_metadata con tokens)
    #   - 'parsing_error': excepción si el parseo falló (None si fue exitoso)
    #
    # Esto nos permite capturar el token usage sin perder el structured output.
    # ------------------------------------------------------------------
    llm = ChatOpenAI(model=AGENT_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(ContractChangeOutput, include_raw=True)
    chain = prompt | structured_llm

    start_time = time.time()

    # ------------------------------------------------------------------
    # Ejecutar la chain dentro de un span de Langfuse v4
    # ------------------------------------------------------------------
    with langfuse.start_as_current_observation(
        as_type="span",
        name="extraction_agent",
        input={
            "model":                  AGENT_MODEL,
            "context_map_preview":    context_map[:300] + "...",
            "text_original_preview":  text_original[:200] + "...",
            "text_amendment_preview": text_amendment[:200] + "...",
        },
    ):
        try:
            # Invocar la chain: devuelve dict {parsed, raw, parsing_error}
            raw_output = chain.invoke({
                "context_map":    context_map,
                "text_original":  text_original,
                "text_amendment": text_amendment,
            })

        except OutputParserException as e:
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"Error de parseo en Agente 2: {e}",
            )
            raise ExtractionError(
                f"❌ El Agente de Extracción recibió una respuesta que no pudo estructurar.\n"
                f"   Aseguráte de que el modelo tiene suficiente contexto para producir el JSON requerido.\n"
                f"   Detalle técnico: {e}"
            ) from e

        except Exception as e:
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"Error en Agente 2: {type(e).__name__}: {e}",
            )
            raise ExtractionError(
                f"❌ Error en el Agente de Extracción ({type(e).__name__}):\n"
                f"   {e}"
            ) from e

        # ------------------------------------------------------------------
        # Verificar error de parseo devuelto por include_raw=True
        # ------------------------------------------------------------------
        parsing_error = raw_output.get("parsing_error")
        if parsing_error:
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"Validación Pydantic fallida: {parsing_error}",
            )
            raise ExtractionError(
                f"❌ El output del Agente 2 no cumple el schema Pydantic esperado.\n"
                f"   Campos requeridos: sections_changed (lista), topics_touched (lista), summary_of_the_change (texto).\n"
                f"   Detalle técnico: {parsing_error}"
            )

        # Extraer el objeto Pydantic validado
        result: ContractChangeOutput = raw_output.get("parsed")

        # Verificación defensiva: asegurarse de que result no es None
        if result is None:
            langfuse.update_current_span(
                level="ERROR",
                status_message="El Agente 2 devolvio un resultado nulo inesperadamente",
            )
            raise ExtractionError(
                "❌ El Agente de Extracción devolvió un resultado nulo. "
                "Intentá volver a ejecutar el pipeline."
            )

        latency_ms = int((time.time() - start_time) * 1000)

        # Extraer token usage del AIMessage crudo
        ai_message = raw_output.get("raw")
        token_usage = {}
        if ai_message and hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:
            token_usage = {
                "input_tokens":  ai_message.usage_metadata.get("input_tokens"),
                "output_tokens": ai_message.usage_metadata.get("output_tokens"),
                "total_tokens":  ai_message.usage_metadata.get("total_tokens"),
            }

        # Registrar output estructurado, latencia y token usage en el span activo
        langfuse.update_current_span(
            output={
                "sections_changed": result.sections_changed,
                "topics_touched":   result.topics_touched,
                "summary_preview":  result.summary_of_the_change[:300] + "...",
            },
            metadata={
                "model":       AGENT_MODEL,
                "latency_ms":  latency_ms,
                "token_usage": token_usage,
            },
        )

    return result, token_usage
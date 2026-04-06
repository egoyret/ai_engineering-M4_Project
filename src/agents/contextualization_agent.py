"""
agents/contextualization_agent.py
----------------------------------
Agente 1: Contextualización

Responsabilidad: recibir los textos del contrato original y la enmienda,
y producir un "mapa contextual" que describa:
  - Qué secciones existen en cada documento
  - Cómo se corresponden entre sí
  - Cuál es el propósito general de cada bloque

Este agente NO extrae cambios; solo construye el contexto que
necesita el Agente 2 para hacer su trabajo con precisión.

Implementado con LangChain LCEL:
  prompt | llm | StrOutputParser

La observabilidad usa Langfuse v4 (API basada en OpenTelemetry).
"""

import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from langfuse import Langfuse

from src.exceptions import ContextualizationError

# Modelo de texto para este agente (económico, sin necesidad de Vision)
AGENT_MODEL = "gpt-4o-mini"



def run_contextualization_agent(
    text_original: str,
    text_amendment: str,
    langfuse: Langfuse,
) -> tuple:  # (context_map: str, token_usage: dict)
    """
    Ejecuta el Agente 1: analiza y compara la estructura de ambos documentos.

    Crea un span hijo en la traza activa de Langfuse (v4) registrando
    el input, output y latencia de la ejecución.

    Args:
        text_original:   Texto extraído del contrato original (por image_parser).
        text_amendment:  Texto extraído de la enmienda/adenda (por image_parser).
        langfuse:        Instancia de Langfuse para registrar observaciones.

    Returns:
        Tupla (context_map, token_usage):
          - context_map: mapa contextual como string de texto estructurado.
          - token_usage: dict con input_tokens, output_tokens, total_tokens.
    """
    # ------------------------------------------------------------------
    # Definir el prompt del Agente 1
    # ------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Eres un analista legal especializado en comparación de documentos contractuales.
Tu tarea es analizar la ESTRUCTURA de un contrato original y su enmienda/adenda.

NO debes identificar cambios específicos todavía. Solo debes:
1. Listar las secciones/cláusulas que existen en el contrato ORIGINAL, con una breve descripción de su propósito.
2. Listar las secciones/cláusulas que existen en la ENMIENDA, con una breve descripción de su propósito.
3. Indicar cómo se corresponden entre sí: qué secciones de la enmienda reemplazan, modifican o complementan al original.
4. Describir el propósito general de cada bloque del documento.
5. Si los objectos de los contratos son totalmente distintos, indica que no se puede realizar el análisis. El objeto de un contrato lo puedes obtener del titulo y/o de las primeras lineas del texto del mismo.

Produce un mapa contextual claro y bien organizado que sirva como guía para el análisis posterior.
Responde en el mismo idioma de los documentos."""
        ),
        (
            "human",
            """=== CONTRATO ORIGINAL ===
{text_original}

=== ENMIENDA / ADENDA ===
{text_amendment}

Por favor, produce el mapa contextual comparado de estos dos documentos."""
        ),
    ])

    # ------------------------------------------------------------------
    # Construir la chain con LCEL en dos pasos para capturar token usage.
    #
    # En lugar de: prompt | llm | output_parser  (perdemos el AIMessage)
    # Hacemos:     1. ai_message = (prompt | llm).invoke({...})
    #              2. context_map = output_parser.invoke(ai_message)
    # Esto nos permite leer ai_message.usage_metadata con los tokens usados.
    # ------------------------------------------------------------------
    llm = ChatOpenAI(model=AGENT_MODEL, temperature=0)
    output_parser = StrOutputParser()
    llm_chain = prompt | llm  # chain sin parser final

    start_time = time.time()

    # ------------------------------------------------------------------
    # Ejecutar la chain dentro de un span de Langfuse v4
    # ------------------------------------------------------------------
    with langfuse.start_as_current_observation(
        as_type="span",
        name="contextualization_agent",
        input={
            "model":                  AGENT_MODEL,
            "text_original_preview":  text_original[:300] + "...",
            "text_amendment_preview": text_amendment[:300] + "...",
        },
    ):
        try:
            # Paso 1: invocar el LLM y obtener el AIMessage (con usage_metadata)
            ai_message = llm_chain.invoke({
                "text_original":  text_original,
                "text_amendment": text_amendment,
            })

            # Paso 2: parsear el AIMessage a string con el StrOutputParser
            context_map: str = output_parser.invoke(ai_message)

        except OutputParserException as e:
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"Error de parseo en Agente 1: {e}",
            )
            raise ContextualizationError(
                f"❌ El Agente de Contextualización recibió una respuesta que no pudo procesar.\n"
                f"   Detalle técnico: {e}"
            ) from e

        except Exception as e:
            # Captura errores de OpenAI via LangChain (rate limit, auth, timeout, etc.)
            langfuse.update_current_span(
                level="ERROR",
                status_message=f"Error en Agente 1: {type(e).__name__}: {e}",
            )
            raise ContextualizationError(
                f"❌ Error en el Agente de Contextualización ({type(e).__name__}):\n"
                f"   {e}"
            ) from e

        latency_ms = int((time.time() - start_time) * 1000)

        # Extraer token usage del AIMessage (disponible en LangChain >= 0.2)
        token_usage = {}
        if hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:
            token_usage = {
                "input_tokens":  ai_message.usage_metadata.get("input_tokens"),
                "output_tokens": ai_message.usage_metadata.get("output_tokens"),
                "total_tokens":  ai_message.usage_metadata.get("total_tokens"),
            }

        # Registrar output, latencia y token usage en el span activo
        langfuse.update_current_span(
            output={"context_map_preview": context_map[:500] + "..."},
            metadata={
                "model":       AGENT_MODEL,
                "latency_ms":  latency_ms,
                "token_usage": token_usage,
            },
        )

    return context_map, token_usage

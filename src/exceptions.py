"""
exceptions.py
-------------
Jerarquía de excepciones personalizadas del sistema de análisis de contratos.

Usar excepciones tipadas permite:
  - Distinguir el origen exacto del error (parsing de imagen, agente, validación)
  - Proveer mensajes claros y accionables al usuario
  - Hacer catching selectivo en main.py sin atrapar excepciones no relacionadas
"""


class ContractPipelineError(Exception):
    """
    Excepción base del pipeline.
    Todas las excepciones del sistema heredan de esta clase, lo que permite
    hacer un catch genérico en main.py si se desea.
    """
    pass


class ImageParsingError(ContractPipelineError):
    """
    Error durante el parsing de una imagen con GPT-4o Vision.

    Causas típicas:
      - Archivo de imagen no encontrado o corrupto
      - Error de autenticación con la API de OpenAI
      - Rate limit o timeout de la API
      - Respuesta vacía o inválida del modelo
    """
    pass


class ContextualizationError(ContractPipelineError):
    """
    Error durante la ejecución del Agente 1 (Contextualización).

    Causas típicas:
      - Error de la API de OpenAI vía LangChain
      - Respuesta vacía o inesperada del LLM
    """
    pass


class ExtractionError(ContractPipelineError):
    """
    Error durante la ejecución del Agente 2 (Extracción de cambios).

    Causas típicas:
      - Error de la API de OpenAI vía LangChain
      - Fallo al parsear el JSON estructurado
      - Validación Pydantic fallida (schema no cumplido)
    """
    pass

class BadContractsError(ContractPipelineError):
    """
    Error durante la ejecución del Agente 2 (Extracción de cambios).

    Caso de par de contratos qu eno osn comparables

    """
    pass  


class OutputSaveError(ContractPipelineError):
    """
    Error al guardar los archivos de salida en disco.

    Causas típicas:
      - Permisos insuficientes en el directorio output/
      - Disco lleno
    """
    pass

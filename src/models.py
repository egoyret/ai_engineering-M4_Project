"""
models.py
---------
Define los modelos Pydantic que validan y estructuran
el output final del pipeline de análisis de contratos.
"""

from typing import List
from pydantic import BaseModel, Field


class ContractChangeOutput(BaseModel):
    """
    Schema de salida del sistema multi-agente.
    
    Este modelo es usado tanto como instrucción al LLM (structured output)
    como para la validación final del JSON producido por el Agente 2.
    """

    sections_changed: List[str] = Field(
        description=(
            "Lista de identificadores o nombres de las secciones del contrato "
            "que fueron modificadas por la enmienda. "
            "Ejemplo: ['Cláusula 3', 'Artículo 7', 'Numeral 2.1']"
        )
    )

    topics_touched: List[str] = Field(
        description=(
            "Lista de categorías legales o comerciales afectadas por los cambios. "
            "Ejemplo: ['plazos de entrega', 'penalidades', 'precio', 'vigencia']"
        )
    )

    summary_of_the_change: str = Field(
        description=(
            "Descripción clara y detallada de todos los cambios introducidos "
            "por la enmienda respecto al contrato original. "
            "Debe mencionar adiciones, eliminaciones y/o modificaciones."
        )
    )

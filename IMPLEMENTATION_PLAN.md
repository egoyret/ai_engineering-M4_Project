# Sistema Multi-Agente de Análisis de Contratos — Plan de Implementación

## Descripción

Sistema que recibe imágenes escaneadas de un **contrato original** y su **adenda/enmienda**, utiliza GPT-4o (Vision) para extraer el texto, y luego pasa esos textos por un pipeline de dos agentes LangChain colaborativos:
1. **ContextualizationAgent** → construye un mapa de estructura comparada entre ambos documentos
2. **ExtractionAgent** → identifica y describe cada cambio (adiciones, eliminaciones, modificaciones)

La salida es un JSON validado por Pydantic, con trazabilidad completa en Langfuse.

---

## Estructura de archivos propuesta

```
ai_engineering-M4_Project/
├── .env                         # Keys existentes (OpenAI + Langfuse)
├── data/test_contracts/         # Imágenes de prueba (ya existen)
└── src/
    ├── __init__.py
    ├── main.py                  # [MODIFY] Punto de entrada del pipeline completo
    ├── models.py                # [MODIFY] Modelos Pydantic (ContractChangeOutput)
    ├── image_parser.py          # [OVERWRITE] parse_contract_image() con Langfuse spans
    └── agents/
        ├── __init__.py
        ├── contextualization_agent.py  # [MODIFY] Agente 1
        └── extraction_agent.py         # [MODIFY] Agente 2
```

---

## Decisiones técnicas

### Stack confirmado
| Componente | Implementación |
|---|---|
| Vision/Parsing | OpenAI `client.responses.create` con `gpt-4.1` (ya probado en image_parser.py) |
| Agentes | **LangChain** (LCEL chains con `ChatOpenAI`) |
| Validación | **Pydantic v2** `model_validate()` |
| Observabilidad | **Langfuse** SDK directo (`langfuse.trace()` + `.span()`) |
| Env vars | `python-dotenv` |

> [!IMPORTANT]
> En el `.env` existe `OPENAI_MODEL=gpt-4o-mini`. Para el parsing de imágenes se usará `gpt-4.1` (como en image_parser.py existente) ya que tiene capacidad Vision superior. Para los agentes LangChain se puede usar `gpt-4o-mini` (más económico).

---

## Cambios propuestos por archivo

### `src/models.py` — [MODIFY]
Define el modelo Pydantic de salida:
```python
class ContractChangeOutput(BaseModel):
    sections_changed: List[str]   # secciones modificadas
    topics_touched: List[str]     # categorías legales/comerciales afectadas
    summary_of_the_change: str    # descripción detallada
```

---

### `src/image_parser.py` — [OVERWRITE]
Función `parse_contract_image(image_path, langfuse_trace)`:
- Codifica imagen a base64
- Llama a `client.responses.create` con GPT-4.1 Vision
- Crea un **span hijo de Langfuse** con input (path), output (texto extraído), latencia y tokens
- Retorna el texto extraído como `str`

---

### `src/agents/contextualization_agent.py` — [MODIFY]
Clase/función `ContextualizationAgent`:
- Recibe: `text_original: str`, `text_amendment: str`, `langfuse_trace`
- Usa una LangChain `ChatOpenAI` chain (LCEL: `prompt | llm | parser`)
- Prompt instruye al modelo a comparar estructura: qué secciones hay en ambos, cómo se corresponden, propósito de cada bloque
- Output: texto estructurado (mapa contextual)
- Crea span Langfuse con input/output/latencia

---

### `src/agents/extraction_agent.py` — [MODIFY]
Clase/función `ExtractionAgent`:
- Recibe: `context_map: str`, `text_original: str`, `text_amendment: str`, `langfuse_trace`
- Usa LangChain chain con `response_format` o structured output para que el LLM devuelva JSON compatible con `ContractChangeOutput`
- Prompt instruye a identificar adiciones, eliminaciones y modificaciones
- Output: objeto `ContractChangeOutput` validado con Pydantic
- Crea span Langfuse con input/output/latencia

---

### `src/main.py` — [MODIFY]
Pipeline principal `run_pipeline(original_path, amendment_path)`:
```
1. Crear trace raíz Langfuse: "contract-analysis"
2. parse_contract_image(original)  → span "parse_original_contract"
3. parse_contract_image(amendment) → span "parse_amendment_contract"
4. ContextualizationAgent(...)     → span "contextualization_agent"
5. ExtractionAgent(...)            → span "extraction_agent"
6. Validar con Pydantic ContractChangeOutput
7. Imprimir JSON validado
8. Finalizar trace Langfuse
```

Al ejecutar `main.py` se procesa el primer par de contratos de prueba (`documento_1`), pero el pipeline acepta cualquier par de rutas como argumento.

---

## Open Questions

> [!IMPORTANT]
> **Modelo para los agentes LangChain**: El `.env` tiene `OPENAI_MODEL=gpt-4o-mini`. Para los agentes (Agente 1 y 2) usaré `gpt-4o-mini` para economizar costos, y `gpt-4.1` solo para el parsing de imágenes. ¿Estás de acuerdo?

> [!NOTE]
> **LangChain vs OpenAI directo**: La consigna especifica LangChain para los agentes. Usaré LCEL (LangChain Expression Language) que es el patrón moderno recomendado. Si prefieres el estilo clásico `AgentExecutor`, avísame.

---

## Plan de verificación

### Automatizado
```bash
# Instalar dependencias
pip install langchain langchain-openai langfuse pydantic python-dotenv

# Ejecutar pipeline con documento_1 de prueba
python src/main.py
```

### Manual
- ✅ Verificar que el JSON de salida cumple el schema Pydantic (`sections_changed`, `topics_touched`, `summary_of_the_change`)
- ✅ Verificar en el dashboard de Langfuse que aparecen los spans: `contract-analysis > parse_original_contract > parse_amendment_contract > contextualization_agent > extraction_agent`
- ✅ Verificar que el output lleva el idioma del documento (español)

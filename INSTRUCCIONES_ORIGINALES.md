🎯 Objetivo
Desarrollar un sistema multi-agente autónomo que procese imágenes escaneadas de un contrato original y su adenda. El sistema deberá extraer el texto utilizando modelos fundacionales multimodales (Visión) y, mediante la colaboración de dos agentes especializados, identificar, extraer y resumir los cambios legales aplicados. El resultado final debe ser un formato estructurado (JSON) estrictamente validado, con trazabilidad completa de cada paso del proceso para garantizar su uso en entornos de producción.

⚙️ Stack técnico
OpenAI GPT-4o (Vision) → para parsear las imágenes de contratos y convertirlas en texto estructurado.
LangChain → para implementar y orquestar los dos agentes colaborativos.
Pydantic → para validar y estructurar el output final del sistema.
Langfuse → para instrumentar el trazado completo del workflow (image parsing, ejecución de agentes, validación).
Python + python-dotenv → como lenguaje base y manejo seguro de variables de entorno.
⚙️ Consigna  resumen
Crear una aplicación en Python que reciba dos imágenes (contrato original y adenda). El sistema deberá usar un LLM Multimodal para leer los documentos y pasarlos por un flujo de dos agentes: uno que entiende el contexto y estructura del documento, y otro que extrae las diferencias. La salida del sistema debe ser un JSON validado por Pydantic que contenga las secciones modificadas, los temas legales afectados y un resumen preciso de los cambios. Todas las llamadas a la API y acciones de los agentes deben quedar registradas en un dashboard de Langfuse.

⚙️ Consigna técnica (pasos o etapas)
Paso 1 - Parsing multimodal de imágenes
Implementar una función parse_contract_image() que reciba el path de una imagen (JPEG/PNG), la codifique en base64 y realice una llamada a la API de GPT-4o con capacidad multimodal. El prompt de visión debe instruir al modelo a extraer el texto completo del contrato de la forma más fiel posible. Este paso debe ejecutarse dos veces: una para el contrato original y otra para la enmienda. La observabilidad de estas ejecuciones (inputs, outputs, latencia y tokens) debe registrarse mediante spans de Langfuse dentro del pipeline principal (main).

Paso 2 - Agente 1: Contextualización
Implementar ContextualizationAgent, cuya responsabilidad es recibir los dos textos parseados y producir un análisis de estructura comparada: identificar qué secciones existen en ambos documentos, cómo se corresponden entre sí y cuál es el propósito general de cada bloque. El output del agente puede ser texto estructurado (no necesariamente JSON) que funcione como mapa contextual para el siguiente agente. Este agente no extrae cambios; solo construye el contexto necesario para el análisis posterior.

Paso 3 – Agente 2: Extracción de cambios
Implementar ExtractionAgent, que recibe como input el output del Agente 1 (el mapa contextual) junto con ambos textos, y tiene la responsabilidad exclusiva de identificar, aislar y describir cada cambio introducido por la enmienda. Debe distinguir entre adiciones, eliminaciones y modificaciones. Su output debe ser un JSON estructurado con los tres campos requeridos, listo para ser validado por Pydantic.

Paso 4 – Validación con Pydantic
Definir el modelo ContractChangeOutput con los campos:

sections_changed: List[str] — identificadores de secciones modificadas.
topics_touched: List[str] — categorías legales/comerciales afectadas.
summary_of_the_change: str — descripción detallada de los cambios.
El output del Agente 2 debe cumplir este schema y ser validado utilizando Pydantic. Esto puede implementarse mediante validación explícita (model_validate()) o mediante structured outputs usando response_format=ContractChangeOutput.

Paso 5 – Trazabilidad con Langfuse
Instrumentar el pipeline con trazabilidad usando Langfuse.

Se recomienda implementar un span raíz que represente la ejecución completa del pipeline y spans hijos para cada etapa principal.

Una posible estructura es:

contract-analysis

├── parse_original_contract

├── parse_amendment_contract

├── contextualization_agent

└── extraction_agent

Cada span debe incluir información relevante como input, output, latencia y metadata útil para debugging o auditoría.


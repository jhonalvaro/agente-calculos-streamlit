# agente-calculos-streamlit

Este repositorio contiene el código para una aplicación de agente de cálculos desplegada en Streamlit Community Cloud.

## Despliegue en Streamlit Cloud

Esta aplicación está diseñada para ser una demostración o un prototipo básico debido a las limitaciones de recursos de Streamlit Community Cloud.

**LIMITACIONES CRÍTICAS (Streamlit Cloud):**
- **RAM**: Solo 1GB (comparado con los 16GB que podría ofrecer una plataforma como HuggingFace Spaces). Esto restringe severamente la complejidad y el tamaño de los modelos o datos que se pueden manejar.
- **Precisión de Cálculo**: Limitada a un máximo de aproximadamente 25 dígitos para operaciones matemáticas (comparado con 50 o más en entornos con más memoria).
- **Métodos Implementables**: Solo una fracción de los métodos de cálculo o validación (aproximadamente 3 de 5) pueden ser implementados de manera estable.
- **Escalabilidad**: No hay opciones de pago para aumentar los recursos. El límite de 3 aplicaciones públicas gratuitas también es una consideración.

**RECOMENDACIÓN**:
Se recomienda usar esta versión de Streamlit principalmente para:
- ✅ Demostración rápida y prototipado.
- ✅ Validación de la interfaz de usuario (UI) y experiencia de usuario (UX).
- ✅ Producto Mínimo Viable (MVP) para obtener feedback inicial.

**NO SE RECOMIENDA** para un agente de producción completo y robusto. Para tales casos, se sugiere considerar plataformas con más recursos como HuggingFace Spaces, que pueden ofrecer hasta 16GB de RAM y permitir una funcionalidad completa.

Consulte el archivo `streamlit_app.py` para ver la implementación específica y las advertencias dentro de la propia aplicación.

Para desplegar su propia instancia, siga las instrucciones detalladas en el issue que originó este código. Necesitará una API Key de Gemini y seguir los pasos de configuración del repositorio y Streamlit Cloud.

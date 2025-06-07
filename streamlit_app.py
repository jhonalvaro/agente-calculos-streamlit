import streamlit as st

st.title("Agente de Cálculos (Versión Streamlit)")

st.warning("Esta es una versión limitada desplegada en Streamlit Cloud.")

st.info(
    """
    **LIMITACIONES CRÍTICAS (Streamlit Cloud):**
    - RAM: Solo 1GB (vs 16GB HuggingFace)
    - Precisión: Máximo 25 dígitos (vs 50)
    - Métodos: Solo 3 de 5 implementables
    - Escalabilidad: Sin tier pagado disponible
    """
)

st.markdown(
    """
    **RECOMENDACIÓN**: Usar Streamlit solo para:
    - ✅ Demo/prototipo básico
    - ✅ Validación de UI/UX
    - ✅ MVP para obtener feedback
    - ❌ **NO para agente completo de producción**

    Para la versión completa y con mayor capacidad, por favor considere la versión desplegada en HuggingFace Spaces.
    """
)

# Placeholder for potential future basic functionality
st.header("Funcionalidad Básica")
number_input = st.number_input("Ingrese un número (para demostración)", value=0.0, step=0.1, format="%.2f")
st.write(f"Número ingresado: {number_input}")

st.sidebar.header("Acerca de")
st.sidebar.info("Este agente realiza cálculos con precisión ajustable. Esta versión de Streamlit es para demostración y tiene funcionalidad limitada.")

st.balloons()

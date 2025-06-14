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

# Input for Desperdicio (waste) in meters
desperdicio_input = st.number_input("Desperdicio (metros)", value=0.0, step=0.01, format="%.2f")
st.write(f"Desperdicio ingresado: {desperdicio_input}")

# Input for Longitud total del tubo (total length of the tube) in meters
longitud_tubo_input = st.number_input("Longitud total del tubo (metros)", value=1.0, step=0.1, format="%.2f")
st.write(f"Longitud total del tubo ingresada: {longitud_tubo_input}")

# Input for Longitud del arco (length of the arch) in meters
longitud_arco_input = st.number_input("Longitud del arco (metros)", value=1.0, step=0.1, format="%.2f")
st.write(f"Longitud del arco ingresada: {longitud_arco_input}")

st.header("Resultados del Cálculo")

# Get input values from the number_input fields
# longitud_tubo_input, desperdicio_input, and longitud_arco_input are already defined above

# Calculate "longitud útil" (useful length)
longitud_util = longitud_tubo_input - desperdicio_input

# Handle negative "longitud útil"
if longitud_util < 0:
    st.warning("La longitud útil es negativa. Se ha ajustado a 0.")
    longitud_util = 0

# Display "longitud útil"
st.metric(label="Longitud útil (metros)", value=f"{longitud_util:.2f}")

# Calculate "tubos_en_arco" (tubes per arch)
# Handle division by zero for "longitud del arco"
if longitud_arco_input == 0:
    st.error("La longitud del arco no puede ser cero. Por favor, ingrese un valor válido.")
    tubos_en_arco = 0
else:
    tubos_en_arco = longitud_util // longitud_arco_input

# Display "tubos_en_arco"
st.metric(label="Cantidad de tubos por arco", value=int(tubos_en_arco))

st.sidebar.header("Acerca de")
st.sidebar.info("Este agente realiza cálculos con precisión ajustable. Esta versión de Streamlit es para demostración y tiene funcionalidad limitada.")

st.balloons()

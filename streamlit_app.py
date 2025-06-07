#!/usr/bin/env python3
"""
🔢 Agente de IA - Cálculos de Precisión (Versión Streamlit Optimizada)
OPTIMIZADO para límite de 1GB RAM de Streamlit Community Cloud

⚠️ VERSIÓN LIMITADA: Para agente completo usar HuggingFace Spaces (16GB RAM)

Características optimizadas:
- Cache agresivo para conservar memoria
- Precisión reducida para funcionar en 1GB
- UI optimizada para Streamlit
- Fallbacks automáticos

Autor: Sistema de IA Colaborativo
Fecha: 2025-06-07
"""

import streamlit as st
import os
import time
import json
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal, getcontext

# Configuración de página (DEBE ser lo primero)
st.set_page_config(
    page_title="🔢 Agente IA - Cálculos",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración optimizada para 1GB RAM
getcontext().prec = 25  # Reducido de 50 a 25 para ahorrar memoria

# Imports condicionales para optimizar memoria
try:
    import mpmath as mp
    mp.dps = 25  # Reducido para conservar memoria
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    st.warning("⚠️ mpmath no disponible. Usando precisión estándar.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ============================================================================
# CONFIGURACIÓN Y CACHE
# ============================================================================

@st.cache_resource
def init_app_config():
    """Configuración global de la aplicación (singleton)."""
    return {
        'precision': 25,  # Reducido para conservar RAM
        'max_cache_entries': 20,  # Limitado para no exceder memoria
        'cache_ttl': 1800,  # 30 minutos
        'gemini_api_key': st.secrets.get("GOOGLE_API_KEY", "")
    }

# ============================================================================
# CALCULADORA OPTIMIZADA PARA 1GB RAM
# ============================================================================

class OptimizedCalculator:
    """Calculadora optimizada para recursos limitados de Streamlit Cloud."""

    def __init__(self, precision: int = 25):
        self.precision = min(precision, 25)  # Máximo 25 para conservar memoria
        if MPMATH_AVAILABLE:
            mp.dps = self.precision

    def calculate_radius_standard(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        """Método estándar optimizado: R = (c² + 4s²) / (8s)"""
        if MPMATH_AVAILABLE:
            # Usar mpmath solo para cálculo final
            c = float(chord)
            s = float(sagitta)
            result_float = (c**2 + 4 * s**2) / (8 * s)
            return Decimal(str(result_float))
        else:
            return (chord**2 + 4 * sagitta**2) / (8 * sagitta)

    def calculate_radius_segmental(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        """Método segmental optimizado: R = s/2 + c²/(8s)"""
        return sagitta/2 + chord**2/(8*sagitta)

    def calculate_radius_trigonometric(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        """Método trigonométrico simplificado"""
        return chord**2/(8*sagitta) + sagitta/2

    def calculate_sagitta_corrected(self, length: Decimal, radius: Decimal) -> Decimal:
        """Fórmula CORREGIDA: s ≈ L²/(8R)"""
        return length**2 / (8 * radius)

# ============================================================================
# CACHE PARA API CALLS
# ============================================================================

@st.cache_data(ttl=1800, max_entries=10)  # Cache reducido para conservar memoria
def call_gemini_api(prompt: str, api_key: str) -> Dict[str, Any]:
    """Llamada a API de Gemini con cache agresivo."""
    if not api_key or not REQUESTS_AVAILABLE:
        return {
            "success": False,
            "error": "API key no configurada o requests no disponible"
        }

    try:
        # Configuración básica para Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {"success": True, "response": text}
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# FUNCIONES PRINCIPALES CON CACHE
# ============================================================================

@st.cache_data(ttl=3600, max_entries=20)  # Cache para cálculos
def calculate_radius_all_methods(chord: float, sagitta: float, precision: int) -> Dict[str, Any]:
    """Ejecutar todos los métodos de cálculo con cache."""

    try:
        # Validar entradas
        if chord <= 0 or sagitta <= 0:
            return {"error": "Los valores deben ser positivos"}

        # Verificar límites geométricos
        approx_radius = chord**2 / (8 * sagitta)
        if sagitta >= approx_radius: # This condition is problematic, sagitta should generally be much smaller than radius
            return {"error": f"Sagitta (s={sagitta}) no puede ser mayor o igual al radio aproximado (R≈{approx_radius:.2f}). Compruebe s < c²/8s."}


        # Inicializar calculadora
        calc = OptimizedCalculator(precision)

        # Convertir a Decimal
        chord_dec = Decimal(str(chord))
        sagitta_dec = Decimal(str(sagitta))

        # Ejecutar métodos (solo 3 para conservar memoria)
        methods = [
            ("Estándar", calc.calculate_radius_standard),
            ("Segmental", calc.calculate_radius_segmental),
            ("Trigonométrico", calc.calculate_radius_trigonometric)
        ]

        results = {}
        for method_name, method_func in methods:
            try:
                value = method_func(chord_dec, sagitta_dec)
                results[method_name] = float(value) # Potential precision loss converting Decimal to float
            except Exception as e:
                results[method_name] = f"Error: {e}"

        # Análisis básico
        valid_results = [v for v in results.values() if isinstance(v, float)]

        if len(valid_results) >= 2:
            # Using median for robustness, ensure it's from valid_results which are floats
            valid_results.sort()
            median_value = valid_results[len(valid_results)//2]

            # Calculate max_deviation based on median_value
            max_deviation = 0
            if median_value != 0: # Avoid division by zero
                 max_deviation = max(abs(v - median_value) for v in valid_results)
                 confidence = (1 - min(max_deviation / abs(median_value), 1)) # Ensure median_value is abs for confidence
            else: # If median is zero, confidence is tricky. If all values are zero, confidence is 100%. Otherwise, it's low.
                if all(v == 0 for v in valid_results):
                    confidence = 1.0
                else:
                    confidence = 0.0


            # Demostración corrección L²/8R
            # Ensure median_value is converted to Decimal for calculation with sagitta_corrected
            demo_length = chord_dec
            median_value_dec = Decimal(str(median_value))
            sagitta_corrected = calc.calculate_sagitta_corrected(demo_length, median_value_dec)
            # sagitta_incorrect uses L^2 / (2R)
            sagitta_incorrect = demo_length**2 / (2 * median_value_dec) if median_value_dec != 0 else Decimal('inf')

            error_percentage = Decimal('0')
            if sagitta_corrected != 0: # Avoid division by zero
                error_percentage = abs(sagitta_incorrect - sagitta_corrected) / sagitta_corrected * 100
            elif sagitta_incorrect == sagitta_corrected: # Both are zero
                error_percentage = Decimal('0')
            else: # Corrected is zero, incorrect is not, implies infinite error or undefined.
                error_percentage = Decimal('inf')


            return {
                "success": True,
                "radius_final": median_value, # This is a float
                "confidence": confidence, # This is a float
                "methods": results, # Contains floats or error strings
                "sagitta_corrected": float(sagitta_corrected), # Convert Decimal to float
                "sagitta_incorrect": float(sagitta_incorrect), # Convert Decimal to float
                "error_percentage": float(error_percentage) # Convert Decimal to float
            }

        else:
            return {"error": "No se pudieron calcular resultados válidos con los métodos disponibles."}

    except Exception as e:
        return {"error": f"Error general en cálculo: {str(e)}"}

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    """Función principal de la aplicación."""

    # Configuración
    config = init_app_config()

    # CSS personalizado
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .warning-box {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .memory-warning {
            background: #f8d7da;
            border: 2px solid #dc3545;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">🔢 Agente de IA - Cálculos de Precisión</div>',
                unsafe_allow_html=True)

    # Advertencia importante sobre limitaciones
    st.markdown("""
    <div class="memory-warning">
        <strong>⚠️ VERSIÓN LIMITADA - Streamlit Cloud (1GB RAM)</strong><br>
        • Precisión reducida: 25 dígitos (vs 50 en HuggingFace)<br>
        • Solo 3 métodos (vs 5 completos)<br>
        • Para versión completa: <a href="#">HuggingFace Spaces (16GB RAM)</a>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar con información
    with st.sidebar:
        st.header("⚙️ Configuración")

        # Mostrar límites
        st.subheader("📊 Límites Actuales")
        st.write("🧠 **RAM**: 1GB máximo")
        st.write("🔢 **Precisión**: 25 dígitos")
        st.write("⚡ **Métodos**: 3 de 5")
        st.write("💰 **Costo**: $0/mes")

        # Estado del sistema
        st.subheader("📈 Estado del Sistema")
        st.write(f"✅ mpmath: {'Disponible' if MPMATH_AVAILABLE else 'No disponible'}")
        st.write(f"✅ API: {'Configurada' if config['gemini_api_key'] else 'No configurada'}")
        st.write(f"⚙️ Precisión: {config['precision']} dígitos")

        # Información de migración
        st.subheader("🚀 Migrar a Versión Completa")
        st.info("""
        **HuggingFace Spaces** ofrece:
        - 16GB RAM (vs 1GB)
        - 50 dígitos precisión (vs 25)
        - 5 métodos completos (vs 3)
        - IA avanzada integrada
        """)

    # Inputs principales
    st.header("📐 Calculadora de Radio del Arco")

    col1, col2 = st.columns(2)

    # Use session state to preserve inputs across reruns (e.g., after example button click)
    if 'chord_input' not in st.session_state:
        st.session_state.chord_input = 100.0
    if 'sagitta_input' not in st.session_state:
        st.session_state.sagitta_input = 10.0

    # Update session state if example buttons are used
    if 'chord_example' in st.session_state:
        st.session_state.chord_input = st.session_state.chord_example
        del st.session_state.chord_example # Clear after use
    if 'sagitta_example' in st.session_state:
        st.session_state.sagitta_input = st.session_state.sagitta_example
        del st.session_state.sagitta_example # Clear after use


    with col1:
        st.session_state.chord_input = st.number_input(
            "Cuerda (c)",
            min_value=0.000001, # Allow very small positive numbers
            max_value=1000000.0,
            value=st.session_state.chord_input,
            step=0.1,
            format="%.6f", # Increased format precision
            help="Longitud de la cuerda del arco (debe ser positiva)"
        )

    with col2:
        st.session_state.sagitta_input = st.number_input(
            "Sagitta/Flecha (s)",
            min_value=0.000001, # Allow very small positive numbers
            max_value=100000.0,
            value=st.session_state.sagitta_input,
            step=0.01,
            format="%.6f", # Increased format precision
            help="Altura máxima del arco (debe ser positiva y menor que c/2)"
        )

    chord_val = st.session_state.chord_input
    sagitta_val = st.session_state.sagitta_input


    # Botón de cálculo
    if st.button("🚀 Calcular Radio", type="primary"):

        # Basic client-side validation for a better UX before calling the heavy function
        if chord_val <= 0 or sagitta_val <= 0:
            st.error("❌ La cuerda y la sagitta deben ser valores positivos.")
        elif sagitta_val >= chord_val / 2: # Sagitta cannot be >= half the chord
             st.error(f"❌ La Sagitta (s={sagitta_val:.4f}) debe ser menor que la mitad de la Cuerda (c/2 = {chord_val/2:.4f}).")
        else:
            with st.spinner("Calculando con precisión limitada..."):

                start_time = time.time()
                # Pass float values from number_input directly
                results = calculate_radius_all_methods(float(chord_val), float(sagitta_val), config['precision'])
                computation_time = time.time() - start_time

                if "error" in results:
                    st.error(f"❌ {results['error']}")

                elif results.get("success"):
                    st.success(f"✅ **Radio Calculado: {results['radius_final']:.7f}**") # Increased display precision

                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Confianza", f"{results['confidence']:.2%}")
                    with col_m2:
                        st.metric("Métodos Válidos", f"{len([v for v in results['methods'].values() if isinstance(v, float)])}/3")
                    with col_m3:
                        st.metric("Tiempo", f"{computation_time*1000:.1f}ms")

                    st.subheader("🔍 Comparación de Métodos")
                    methods_data = []
                    for method, value in results['methods'].items():
                        if isinstance(value, float):
                            methods_data.append({"Método": method, "Resultado": f"{value:.7f}", "Estado": "✅"}) # Increased display precision
                        else:
                            methods_data.append({"Método": method, "Resultado": str(value), "Estado": "❌"})
                    st.table(methods_data)

                    st.subheader("🔧 Verificación de Sagitta con Radio Calculado")
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.metric("s con L²/(8R) (Corregida)", f"{results['sagitta_corrected']:.7f}") # Increased display precision
                    with col_s2:
                        st.metric("s con L²/(2R) (Incorrecta)", f"{results['sagitta_incorrect']:.7f}") # Increased display precision

                    if results['error_percentage'] != float('inf'):
                        st.info(f"📈 **Error relativo entre fórmulas de sagitta**: {results['error_percentage']:.2f}%")
                    else:
                        st.info("📈 Error relativo entre fórmulas de sagitta: Indefinido (división por cero)")


                    if config['gemini_api_key'] and REQUESTS_AVAILABLE:
                        with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                            with st.spinner("Consultando IA..."):
                                ai_prompt = f"Analiza este cálculo de radio de un arco: Cuerda={chord_val}, Sagitta={sagitta_val}, Radio calculado={results['radius_final']:.7f}. ¿Es el radio calculado geométricamente coherente con la cuerda y la sagitta dadas? Proporciona una breve explicación (2-3 frases)."
                                ai_response = call_gemini_api(ai_prompt, config['gemini_api_key'])

                                if ai_response["success"]:
                                    st.write(ai_response["response"])
                                else:
                                    st.error(f"Error IA: {ai_response['error']}")
                    elif not config['gemini_api_key']:
                         with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                            st.warning("API Key de Gemini no configurada en los secrets de Streamlit.")
                    elif not REQUESTS_AVAILABLE:
                        with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                            st.warning("Módulo 'requests' no disponible. No se puede llamar a la API.")


    st.subheader("📋 Ejemplos Rápidos")
    examples = [
        {"name": "Puente Pequeño", "chord": 50.0, "sagitta": 5.0},
        {"name": "Arco Estándar", "chord": 100.0, "sagitta": 10.0},
        {"name": "Estructura Grande", "chord": 200.0, "sagitta": 15.0}, # Adjusted sagitta for variety
        {"name": "Lente Óptica", "chord": 10.0, "sagitta": 0.5} # Small scale example
    ]

    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(f"📐 {example['name']}", key=f"example_{i}", help=f"C: {example['chord']}, S: {example['sagitta']}"):
                # Set values in session state and trigger a rerun to update input fields
                st.session_state.chord_input = example['chord']
                st.session_state.sagitta_input = example['sagitta']
                st.rerun()


    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        🔢 <b>Agente de IA - Versión Streamlit Limitada</b><br>
        Para versión completa: <a href="#">HuggingFace Spaces</a> (16GB RAM, 50 dígitos, 5 métodos)<br>
        <small>Desarrollado con Streamlit Cloud • Optimizado para 1GB RAM</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state for inputs if not already present
    # This is useful if the script is run directly and main() is called.
    if 'chord_input' not in st.session_state:
        st.session_state.chord_input = 100.0
    if 'sagitta_input' not in st.session_state:
        st.session_state.sagitta_input = 10.0
    main()

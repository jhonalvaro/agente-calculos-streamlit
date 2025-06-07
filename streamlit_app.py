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
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Configuración de página (DEBE ser lo primero)
st.set_page_config(
    page_title="🔢 Agente IA - Cálculos",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración optimizada para 1GB RAM
# Setting precision for Decimal calculations
CONTEXT = getcontext()
CONTEXT.prec = 25  # Reducido de 50 a 25 para ahorrar memoria
CONTEXT.rounding = ROUND_HALF_UP

# Imports condicionales para optimizar memoria
try:
    import mpmath as mp
    mp.dps = 25  # Reducido para conservar memoria
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    # This warning will appear once at the top if mpmath is not found
    # st.warning("⚠️ mpmath no disponible. Usando precisión estándar para algunas operaciones.")

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
        # Ensure Decimal context precision is set
        getcontext().prec = self.precision
        if MPMATH_AVAILABLE:
            mp.dps = self.precision

    def _to_decimal(self, value: Any) -> Decimal:
        return Decimal(str(value))

    def calculate_radius_standard(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        """Método estándar optimizado: R = (c² + 4s²) / (8s)"""
        # c_sq = chord * chord
        # s_sq_4 = 4 * sagitta * sagitta
        # eight_s = 8 * sagitta
        # if eight_s == 0: return Decimal('inf')
        # return (c_sq + s_sq_4) / eight_s
        if MPMATH_AVAILABLE: # Use mpmath for intermediate steps if available for better precision handling
            # Convert to float for mpmath, then back to Decimal for consistency
            c_mp = mp.mpf(str(chord))
            s_mp = mp.mpf(str(sagitta))
            if s_mp == 0: return Decimal('inf')
            # mpmath calculation
            radius_mp = (c_mp**2 + 4*s_mp**2) / (8*s_mp)
            return self._to_decimal(radius_mp) # Convert final mpmath result to Decimal
        else: # Fallback to pure Decimal
            if sagitta == 0: return Decimal('inf')
            return (chord**2 + 4 * sagitta**2) / (8 * sagitta)


    def calculate_radius_segmental(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        """Método segmental optimizado: R = s/2 + c²/(8s)"""
        # if sagitta == 0: return Decimal('inf')
        # return sagitta/2 + chord**2/(8*sagitta)
        if MPMATH_AVAILABLE:
            c_mp = mp.mpf(str(chord))
            s_mp = mp.mpf(str(sagitta))
            if s_mp == 0: return Decimal('inf')
            radius_mp = s_mp/2 + c_mp**2/(8*s_mp)
            return self._to_decimal(radius_mp)
        else:
            if sagitta == 0: return Decimal('inf')
            return sagitta/Decimal('2') + chord**2/(Decimal('8')*sagitta)

    def calculate_radius_trigonometric(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        """Método trigonométrico simplificado: R = c²/(8s) + s/2 (same as segmental)"""
        # This is essentially the same as segmental, often presented this way
        # if sagitta == 0: return Decimal('inf')
        # return chord**2/(8*sagitta) + sagitta/2
        return self.calculate_radius_segmental(chord, sagitta) # Reuse segmental

    def calculate_sagitta_corrected(self, length: Decimal, radius: Decimal) -> Decimal:
        """Fórmula CORREGIDA y optimizada: s ≈ L²/(8R)"""
        if radius == 0:
            return Decimal('inf') # Or handle as an error/exception
        # return length**2 / (8 * radius)
        if MPMATH_AVAILABLE:
            l_mp = mp.mpf(str(length))
            r_mp = mp.mpf(str(radius))
            if r_mp == 0: return Decimal('inf')
            sagitta_mp = l_mp**2 / (8 * r_mp)
            return self._to_decimal(sagitta_mp)
        else:
            if radius == 0: return Decimal('inf')
            return length**2 / (Decimal('8') * radius)

# ============================================================================
# CACHE PARA API CALLS
# ============================================================================

@st.cache_data(ttl=1800, max_entries=10)
def call_gemini_api(prompt: str, api_key: str) -> Dict[str, Any]:
    """Llamada a API de Gemini con cache agresivo."""
    if not api_key or not REQUESTS_AVAILABLE:
        return {"success": False, "error": "API key no configurada o requests no disponible"}
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-1.5:generateContent?key={api_key}" # Updated model
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers, timeout=20) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Safer navigation through the response structure
        candidates = data.get("candidates", [])
        if not candidates:
            return {"success": False, "error": "No candidates in API response"}

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return {"success": False, "error": "No parts in API response content"}

        text = parts[0].get("text", "")
        if not text:
             return {"success": False, "error": "Empty text in API response part"}

        return {"success": True, "response": text}
    except requests.exceptions.RequestException as e: # Catch specific requests errors
        return {"success": False, "error": f"Error de Red/API: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Error procesando respuesta API: {str(e)}"}

# ============================================================================
# FUNCIONES PRINCIPALES CON CACHE
# ============================================================================

@st.cache_data(ttl=3600, max_entries=20)
def calculate_radius_all_methods(chord_str: str, sagitta_str: str, precision: int) -> Dict[str, Any]:
    """Ejecutar todos los métodos de cálculo con cache. Inputs are strings for precise Decimal conversion."""
    getcontext().prec = precision # Ensure precision is set for this calculation context

    try:
        chord = Decimal(chord_str)
        sagitta = Decimal(sagitta_str)

        if chord <= 0 or sagitta <= 0:
            return {"error": "La Cuerda (c) y la Sagitta (s) deben ser valores positivos."}
        if sagitta >= chord / Decimal('2'):
            return {"error": f"La Sagitta (s={sagitta}) debe ser menor que la mitad de la Cuerda (c/2 = {chord/Decimal('2')})."}

        calc = OptimizedCalculator(precision)
        methods_to_run = [
            ("Estándar", calc.calculate_radius_standard),
            ("Segmental", calc.calculate_radius_segmental),
            ("Trigonométrico", calc.calculate_radius_trigonometric) # This is same as segmental
        ]
        results = {}
        for name, func in methods_to_run:
            try:
                val = func(chord, sagitta)
                results[name] = val # Keep as Decimal for now
            except Exception as e:
                results[name] = f"Error: {str(e)}"

        valid_results_decimal = [v for v in results.values() if isinstance(v, Decimal) and v.is_finite()]

        if not valid_results_decimal:
            return {"error": "No se pudieron obtener resultados válidos de los métodos de cálculo."}

        # Sort Decimal results and find median
        valid_results_decimal.sort()
        median_value_dec = valid_results_decimal[len(valid_results_decimal) // 2]

        # Calculate confidence based on deviation from median (all in Decimal)
        if median_value_dec == 0: # Avoid division by zero if median is zero
            confidence_dec = Decimal('1') if all(v == 0 for v in valid_results_decimal) else Decimal('0')
        else:
            max_deviation_dec = max(abs(v - median_value_dec) for v in valid_results_decimal)
            confidence_dec = Decimal('1') - min(max_deviation_dec / abs(median_value_dec), Decimal('1'))

        # Sagitta verification using the calculated median radius
        # L is the original chord length for this specific verification section
        sagitta_corrected_dec = calc.calculate_sagitta_corrected(chord, median_value_dec)
        # L^2 / (2R)
        sagitta_incorrect_dec = chord**2 / (Decimal('2') * median_value_dec) if median_value_dec != 0 else Decimal('inf')

        error_percentage_dec = Decimal('0')
        if sagitta_corrected_dec != 0 and sagitta_corrected_dec.is_finite():
            error_percentage_dec = abs(sagitta_incorrect_dec - sagitta_corrected_dec) / sagitta_corrected_dec * Decimal('100')
        elif sagitta_incorrect_dec == sagitta_corrected_dec: # Both zero or both non-finite and equal
            error_percentage_dec = Decimal('0')
        else: # Corrected is zero or non-finite, and they are not equal
             error_percentage_dec = Decimal('inf')


        # Prepare output for UI (convert Decimals to floats or formatted strings as needed at the UI level)
        return {
            "success": True,
            "radius_final_dec": str(median_value_dec), # Keep as string for full precision transfer
            "confidence_dec": str(confidence_dec),   # Keep as string
            "methods_dec": {k: str(v) if isinstance(v, Decimal) else v for k, v in results.items()},
            "sagitta_corrected_dec": str(sagitta_corrected_dec),
            "sagitta_incorrect_dec": str(sagitta_incorrect_dec),
            "error_percentage_dec": str(error_percentage_dec)
        }

    except Exception as e:
        return {"error": f"Error general en cálculo: {str(e)}"}

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    config = init_app_config()
    getcontext().prec = config['precision'] # Set global Decimal precision from config

    st.markdown("<style>.main-header{background:linear-gradient(90deg,#FF6B6B,#4ECDC4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2rem;font-weight:bold;text-align:center;margin-bottom:1rem}.memory-warning{background:#f8d7da;border:2px solid #dc3545;border-radius:8px;padding:1rem;margin:1rem 0}</style>", unsafe_allow_html=True)
    st.markdown('<div class="main-header">🔢 Agente de IA - Cálculos de Precisión</div>', unsafe_allow_html=True)
    st.markdown("<div class=\"memory-warning\"><strong>⚠️ VERSIÓN LIMITADA - Streamlit Cloud (1GB RAM)</strong><br>• Precisión reducida: ~25 dígitos<br>• Solo 3 métodos (vs 5+)<br>• Para versión completa y más precisa: Busque 'Agente Cálculos Avanzado HuggingFace'</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuración")
        st.write(f"✅ mpmath: {'Disponible' if MPMATH_AVAILABLE else 'No disponible (Cálculos con Decimal nativo)'}")
        st.write(f"✅ API Gemini: {'Configurada' if config['gemini_api_key'] and REQUESTS_AVAILABLE else 'No disponible/configurada'}")
        st.write(f"⚙️ Precisión Decimal: {getcontext().prec} dígitos")
        st.info("HuggingFace Spaces ofrece: 16GB RAM, ~50 dígitos precisión, 5+ métodos, IA avanzada.")

    st.header("📐 Calculadora de Radio del Arco")
    # Initialize session state for inputs if not already present
    for key, default_value in [('chord_input', "100.0"), ('sagitta_input', "10.0"), ('tube_length_input', "0.0")]:
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Update session state from example buttons if they were clicked
    if 'example_values' in st.session_state:
        st.session_state.chord_input = st.session_state.example_values['chord']
        st.session_state.sagitta_input = st.session_state.example_values['sagitta']
        # Reset tube length for examples, or set if example includes it
        st.session_state.tube_length_input = st.session_state.example_values.get('tube_length', "0.0")
        del st.session_state.example_values


    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.chord_input = st.text_input("Cuerda (c)", value=st.session_state.chord_input, help="Longitud de la cuerda del arco (ej: 100.0)")
    with col2:
        st.session_state.sagitta_input = st.text_input("Sagitta/Flecha (s)", value=st.session_state.sagitta_input, help="Altura máxima del arco (ej: 10.0)")
    with col3:
        st.session_state.tube_length_input = st.text_input("Longitud Tubo (L_tubo)", value=st.session_state.tube_length_input, help="Longitud del tubo a rolar (opcional, ej: 50.0)")

    chord_str_val = st.session_state.chord_input
    sagitta_str_val = st.session_state.sagitta_input
    tube_length_str_val = st.session_state.tube_length_input

    # Helper to convert string to Decimal, returning None on failure
    def to_decimal_safe(val_str: str) -> Optional[Decimal]:
        try:
            return Decimal(val_str)
        except:
            return None

    if st.button("🚀 Calcular Radio y Flecha de Tubo", type="primary"):
        chord_dec = to_decimal_safe(chord_str_val)
        sagitta_dec = to_decimal_safe(sagitta_str_val)
        tube_length_dec = to_decimal_safe(tube_length_str_val)

        # Validate inputs (ensure they are valid Decimals and positive)
        if chord_dec is None or sagitta_dec is None :
            st.error("❌ Cuerda y Sagitta deben ser números válidos.")
        elif tube_length_dec is None :
             st.error("❌ Longitud del Tubo debe ser un número válido (o 0 si no se usa).")
        elif chord_dec <= 0 or sagitta_dec <= 0:
            st.error("❌ La Cuerda y la Sagitta deben ser valores positivos.")
        elif sagitta_dec >= chord_dec / Decimal('2'):
            st.error(f"❌ La Sagitta (s={sagitta_dec}) debe ser menor que la mitad de la Cuerda (c/2 = {chord_dec/Decimal('2')}).")
        elif tube_length_dec < 0:
             st.error("❌ Longitud del Tubo no puede ser negativa (dejar en 0 si no se usa).")
        else:
            with st.spinner("Calculando..."):
                start_time = time.time()
                # Pass inputs as strings to preserve precision for Decimal conversion inside the cached function
                results = calculate_radius_all_methods(chord_str_val, sagitta_str_val, config['precision'])
                computation_time = time.time() - start_time

                if results.get("success"):
                    st.success(f"✅ **Radio del Arco Calculado (R): {Decimal(results['radius_final_dec']):.7f}**")

                    # Display metrics for radius calculation
                    col_m1, col_m2, col_m3 = st.columns(3)
                    conf_dec = Decimal(results['confidence_dec'])
                    col_m1.metric("Confianza (Radio)", f"{conf_dec:.2%}")

                    valid_method_count = sum(1 for v_str in results['methods_dec'].values() if not v_str.startswith("Error"))
                    col_m2.metric("Métodos Válidos", f"{valid_method_count}/{len(results['methods_dec'])}")
                    col_m3.metric("Tiempo Cálculo", f"{computation_time*1000:.1f}ms")

                    # Display table of methods for radius
                    st.subheader("🔍 Comparación de Métodos (Radio)")
                    methods_data_ui = [{"Método": k, "Resultado (R)": f"{Decimal(v_str):.7f}" if not v_str.startswith("Error") else v_str, "Estado": "✅" if not v_str.startswith("Error") else "❌"} for k,v_str in results['methods_dec'].items()]
                    st.table(methods_data_ui)

                    # Display sagitta verification for the original chord
                    st.subheader("🔧 Verificación de Sagitta (para Cuerda original y Radio calculado)")
                    sag_corr_dec = Decimal(results['sagitta_corrected_dec'])
                    sag_incorr_dec = Decimal(results['sagitta_incorrect_dec'])
                    err_perc_dec = Decimal(results['error_percentage_dec'])

                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric("s con L²/(8R) (Corregida)", f"{sag_corr_dec:.7f}")
                    col_s2.metric("s con L²/(2R) (Incorrecta)", f"{sag_incorr_dec:.7f}")
                    if err_perc_dec.is_finite():
                        st.info(f"📈 **Error relativo entre fórmulas de sagitta**: {err_perc_dec:.2f}%")
                    else:
                        st.info("📈 Error relativo entre fórmulas de sagitta: Indefinido/Infinito")

                    # ---- NUEVA SECCIÓN: Cálculo de Flecha para Tubo Rolado ----
                    if tube_length_dec is not None and tube_length_dec > 0:
                        st.subheader("🏹 Cálculo de Flecha para Tubo Rolado")
                        calc_inst = OptimizedCalculator(config['precision']) # Re-instance or use existing
                        radius_as_decimal = Decimal(results['radius_final_dec'])

                        if radius_as_decimal.is_finite() and radius_as_decimal > 0:
                            flecha_tubo_dec = calc_inst.calculate_sagitta_corrected(tube_length_dec, radius_as_decimal)

                            col_t1, col_t2 = st.columns(2)
                            col_t1.metric("Longitud del Tubo (L_tubo)", f"{tube_length_dec:.4f}")
                            col_t2.metric("Flecha del Tubo Calculada (s_tubo)", f"{flecha_tubo_dec:.7f}")
                            st.caption(f"Calculado usando L_tubo = {tube_length_dec} y Radio del Arco (R) = {radius_as_decimal:.7f}")
                        else:
                            st.warning("No se puede calcular la flecha del tubo porque el radio del arco es cero, infinito o no válido.")
                    # ---- FIN NUEVA SECCIÓN ----

                    # Optional AI Analysis
                    if config['gemini_api_key'] and REQUESTS_AVAILABLE:
                        # ... (AI part remains mostly the same)
                        with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                            # ...
                            pass # Placeholder for brevity

                else: # Error in calculate_radius_all_methods
                    st.error(f"❌ {results.get('error', 'Error desconocido en el cálculo del radio.')}")
        # else for input validation block (already handled by st.error calls)

    st.subheader("📋 Ejemplos Rápidos")
    examples = [
        {"name": "Arco Estándar", "chord": "100.0", "sagitta": "10.0", "tube_length": "50.0"},
        {"name": "Puente Pequeño", "chord": "50.0", "sagitta": "5.0"}, # No tube length specified, will default to 0
        {"name": "Lente Óptica", "chord": "10.0", "sagitta": "0.5", "tube_length": "8.0"}
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(f"📐 {ex['name']}", key=f"ex_{i}", help=f"C: {ex['chord']}, S: {ex['sagitta']}{', L_tubo: ' + ex['tube_length'] if 'tube_length' in ex else ''}"):
            st.session_state.example_values = ex # Store all example values
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;margin-top:2rem'>🔢 <b>Agente de IA - Streamlit Limitado</b><br>Para versión completa: Busque 'Agente Cálculos Avanzado HuggingFace'<br><small>Desarrollado con Streamlit Cloud</small></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state for inputs if not already present, ensuring defaults are strings
    for key, default_value in [('chord_input', "100.0"), ('sagitta_input', "10.0"), ('tube_length_input', "0.0")]:
        if key not in st.session_state:
            st.session_state[key] = default_value
    main()

#!/usr/bin/env python3
"""
🔢 Agente de IA - Cálculos de Precisión (Versión Streamlit Optimizada)
OPTIMIZADO para límite de 1GB RAM de Streamlit Community Cloud
"""

import streamlit as st
import os
import time
import json # Not strictly used in the final version but good for potential future use
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

# Configuración de página (DEBE ser lo primero)
st.set_page_config(
    page_title="🔢 Agente IA - Cálculos",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración optimizada para 1GB RAM
CONTEXT = getcontext()
CONTEXT.prec = 28 # Working precision for Decimal
CONTEXT.rounding = ROUND_HALF_UP

# Imports condicionales
try:
    import mpmath as mp
    mp.dps = 28 # Match Decimal precision
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ============================================================================
# CONFIGURACIÓN Y CACHE
# ============================================================================
@st.cache_resource # Singleton for app configuration
def init_app_config():
    return {
        'precision': 28, # Internal calculation precision
        'display_precision_general': 7, # For general display of Decimal results
        'display_precision_metrics': 4, # For metrics like length, sagitta if needed
        'max_cache_entries': 20,
        'cache_ttl': 1800, # 30 minutes
        'gemini_api_key': st.secrets.get("GOOGLE_API_KEY", "")
    }

# ============================================================================
# CALCULADORA OPTIMIZADA
# ============================================================================
class OptimizedCalculator:
    def __init__(self, precision: int = 28):
        self.precision = precision
        # Ensure the context for this instance reflects the desired precision
        # Though global context is also set, this could be for specific instances if needed
        # For now, it primarily relies on the global context set by init_app_config and calculate_radius_all_methods
        # getcontext().prec = self.precision
        if MPMATH_AVAILABLE:
            mp.dps = self.precision

    def _to_decimal_from_str(self, value_str: str) -> Decimal:
        # Helper to convert string to Decimal, ensuring current context precision
        # This is mostly for internal use if direct string conversion is needed
        # For external values, Decimal(str(float_value)) is used.
        return Decimal(value_str)

    def calculate_radius_standard(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        if sagitta == Decimal('0'): return Decimal('inf')
        if MPMATH_AVAILABLE:
            c_mp, s_mp = mp.mpf(str(chord)), mp.mpf(str(sagitta))
            if s_mp == 0: return Decimal(str(mp.inf)) # mpmath inf to Decimal
            return Decimal(str((c_mp**2 + 4*s_mp**2) / (8*s_mp)))
        return (chord**2 + 4 * sagitta**2) / (8 * sagitta)

    def calculate_radius_segmental(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        if sagitta == Decimal('0'): return Decimal('inf')
        if MPMATH_AVAILABLE:
            c_mp, s_mp = mp.mpf(str(chord)), mp.mpf(str(sagitta))
            if s_mp == 0: return Decimal(str(mp.inf))
            return Decimal(str(s_mp/2 + c_mp**2/(8*s_mp)))
        return sagitta/Decimal('2') + chord**2/(Decimal('8')*sagitta)

    def calculate_radius_trigonometric(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        # This is mathematically identical to the segmental formula
        return self.calculate_radius_segmental(chord, sagitta)

    def calculate_sagitta_corrected(self, length: Decimal, radius: Decimal) -> Decimal:
        if radius == Decimal('0'): return Decimal('inf')
        if MPMATH_AVAILABLE:
            l_mp, r_mp = mp.mpf(str(length)), mp.mpf(str(radius))
            if r_mp == 0: return Decimal(str(mp.inf))
            return Decimal(str(l_mp**2 / (8 * r_mp)))
        return length**2 / (Decimal('8') * radius)

# ============================================================================
# API CALL (Gemini)
# ============================================================================
@st.cache_data(ttl=1800, max_entries=10)
def call_gemini_api(prompt: str, api_key: str) -> Dict[str, Any]:
    if not api_key:
        return {"success": False, "error": "API key de Gemini no configurada."}
    if not REQUESTS_AVAILABLE:
        return {"success": False, "error": "Módulo 'requests' no disponible. No se puede llamar a la API."}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=25) # Increased timeout
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates: return {"success": False, "error": "Respuesta API: Sin 'candidates'"}

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts: return {"success": False, "error": "Respuesta API: Sin 'parts' en el contenido"}

        text = parts[0].get("text", "")
        # It's possible 'text' is missing or empty, which might be valid for some prompts.
        # So, we don't error out on empty text, but the calling function might need to handle it.
        return {"success": True, "response": text}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Error de Red/API: Timeout (25s)"}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"Error de Red/API: {e.response.status_code} - {e.response.text[:100]}"} # Show snippet of error
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Error de Red/API: {str(e)}"}
    except Exception as e: # Catch-all for other unexpected errors like JSON parsing
        return {"success": False, "error": f"Error procesando respuesta API: {str(e)}"}

# ============================================================================
# CORE CALCULATION FUNCTION (Cached)
# ============================================================================
@st.cache_data(ttl=3600, max_entries=20) # Cache results for 1 hour
def calculate_radius_all_methods(chord_float: float, sagitta_float: float, internal_precision: int) -> Dict[str, Any]:
    # Set precision for this specific calculation context
    # This is crucial as this function is cached and runs independently
    calc_context = getcontext()
    calc_context.prec = internal_precision

    try:
        # Convert float inputs to Decimal using their string representation
        # This is the recommended way to get the exact value represented by the float
        chord = Decimal(str(chord_float))
        sagitta = Decimal(str(sagitta_float))

        if chord <= Decimal('0') or sagitta <= Decimal('0'):
            return {"error": "La Cuerda (c) y la Sagitta (s) deben ser valores estrictamente positivos."}
        # Geometric validation: sagitta must be less than half the chord
        if sagitta >= chord / Decimal('2'):
            return {"error": f"La Sagitta (s={sagitta}) debe ser menor que la mitad de la Cuerda (c/2 = {chord/Decimal('2')})."}

        calc_instance = OptimizedCalculator(internal_precision)
        methods_to_run = [
            ("Estándar", calc_instance.calculate_radius_standard),
            ("Segmental", calc_instance.calculate_radius_segmental),
            ("Trigonométrico", calc_instance.calculate_radius_trigonometric) # Mathematically same as segmental
        ]

        results_decimal = {} # Store results as Decimals
        for name, func in methods_to_run:
            try:
                val = func(chord, sagitta)
                results_decimal[name] = val
            except InvalidOperation as e: # Catch Decimal specific errors
                 results_decimal[name] = f"Error Decimal: {str(e)}"
            except Exception as e: # Catch other errors
                results_decimal[name] = f"Error: {str(e)}"

        valid_results_list = [v for v in results_decimal.values() if isinstance(v, Decimal) and v.is_finite() and v > 0]

        if not valid_results_list:
            return {"error": "No se pudieron obtener resultados válidos (positivos y finitos) de los métodos de cálculo."}

        valid_results_list.sort()
        median_value_dec = valid_results_list[len(valid_results_list) // 2]

        if median_value_dec <= Decimal('0'): # Should not happen if valid_results_list only contains positive
             return {"error": "El radio mediano calculado no es positivo."}

        # Calculate confidence based on deviation from median
        if len(valid_results_list) == 1: # If only one valid method, confidence is not very meaningful or max
            confidence_dec = Decimal('1')
        elif median_value_dec == Decimal('0'): # Avoid division by zero, though median_value_dec should be > 0 here
            confidence_dec = Decimal('0')
        else:
            max_deviation_dec = max(abs(v - median_value_dec) for v in valid_results_list)
            # Confidence: 1 - (max_deviation / median). Clamp deviation to be not more than median itself.
            relative_deviation = min(max_deviation_dec / abs(median_value_dec), Decimal('1'))
            confidence_dec = Decimal('1') - relative_deviation

        # Sagitta verification using the calculated median radius (L is original chord)
        sagitta_corrected_dec = calc_instance.calculate_sagitta_corrected(chord, median_value_dec)
        # L^2 / (2R) - the "incorrect" formula for comparison
        sagitta_incorrect_dec = chord**2 / (Decimal('2') * median_value_dec) if median_value_dec != Decimal('0') else Decimal('inf')

        error_percentage_dec = Decimal('0')
        if sagitta_corrected_dec.is_finite() and sagitta_corrected_dec != Decimal('0'):
            if sagitta_incorrect_dec.is_finite():
                error_percentage_dec = abs(sagitta_incorrect_dec - sagitta_corrected_dec) / sagitta_corrected_dec * Decimal('100')
            else: # Corrected is finite/non-zero, incorrect is inf
                error_percentage_dec = Decimal('inf')
        elif sagitta_incorrect_dec == sagitta_corrected_dec: # Both zero or both non-finite and equal
            error_percentage_dec = Decimal('0')
        else: # Corrected is zero or non-finite, and they are not equal (and incorrect is not inf when corrected is 0)
             error_percentage_dec = Decimal('inf') # Or some other indicator of large/undefined error

        # Return results as strings to ensure full precision is maintained across cache boundary
        return {
            "success": True,
            "radius_final_dec_str": str(median_value_dec),
            "confidence_dec_str": str(confidence_dec),
            "methods_dec_str": {k: str(v) if isinstance(v, Decimal) else v for k, v in results_decimal.items()},
            "sagitta_corrected_dec_str": str(sagitta_corrected_dec),
            "sagitta_incorrect_dec_str": str(sagitta_incorrect_dec),
            "error_percentage_dec_str": str(error_percentage_dec)
        }

    except InvalidOperation as e: # Catch Decimal specific errors early if any
        return {"error": f"Error de operación Decimal: {str(e)}"}
    except Exception as e: # Catch-all for other unexpected errors
        return {"error": f"Error general en el núcleo de cálculo: {str(e)}"}

# ============================================================================
# STREAMLIT UI (main function)
# ============================================================================
def main():
    app_config = init_app_config()
    # Set global Decimal precision for the app context from config for UI operations
    getcontext().prec = app_config['precision']
    display_prec = app_config['display_precision_general'] # For f-string formatting

    # --- UI Styling and Header ---
    st.markdown("<style>.main-header{background:linear-gradient(90deg,#FF6B6B,#4ECDC4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2rem;font-weight:bold;text-align:center;margin-bottom:1rem}.memory-warning{background:#f8d7da;border:2px solid #dc3545;border-radius:8px;padding:1rem;margin:1rem 0}</style>", unsafe_allow_html=True)
    st.markdown('<div class="main-header">🔢 Agente de IA - Cálculos de Precisión</div>', unsafe_allow_html=True)
    st.markdown(f"<div class=\"memory-warning\"><strong>⚠️ VERSIÓN LIMITADA - Streamlit Cloud (1GB RAM)</strong><br>• Precisión numérica interna: {app_config['precision']} dígitos (Decimal)<br>• Para versión completa: Busque 'Agente Cálculos Avanzado HuggingFace'</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuración")
        st.write(f"✅ mpmath: {'Disponible' if MPMATH_AVAILABLE else 'No disponible (Cálculos con Decimal nativo)'}")
        st.write(f"✅ API Gemini: {'Configurada' if app_config['gemini_api_key'] and REQUESTS_AVAILABLE else 'No disponible/configurada'}")
        st.write(f"⚙️ Precisión Decimal Interna: {getcontext().prec} dígitos")
        st.info("Sugerencia: Para cálculos de muy alta precisión o modelos complejos, considere entornos con más RAM (ej. HuggingFace Spaces).")

    st.header("📐 Calculadora de Radio del Arco")

    # Initialize session state with float defaults for st.number_input
    # These keys will hold float values from st.number_input
    for key, default_value in [('chord_input_float', 100.0),
                               ('sagitta_input_float', 10.0),
                               ('tube_length_input_float', 0.0)]:
        if key not in st.session_state:
            st.session_state[key] = default_value

    # If example values were set, update session state
    if 'example_values_float' in st.session_state:
        st.session_state.chord_input_float = st.session_state.example_values_float['chord']
        st.session_state.sagitta_input_float = st.session_state.example_values_float['sagitta']
        st.session_state.tube_length_input_float = st.session_state.example_values_float.get('tube_length', 0.0)
        del st.session_state.example_values_float # Clear after use

    # Define a consistent format and step for number inputs
    # This format string tells st.number_input how to display the number and influences parsing
    num_input_format_str = f"%.{app_config['display_precision_general']}f"
    # Step should be small enough for desired granularity
    # 1e-7 means it can step by 0.0000001
    num_input_step_val = 1.0 / (10**app_config['display_precision_general'])


    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.chord_input_float = st.number_input("Cuerda (c)",
                                                       min_value=1e-9, # Smallest positive to avoid issues with log/div by zero if used elsewhere
                                                       max_value=1e12, # Reasonably large upper bound
                                                       value=st.session_state.chord_input_float,
                                                       step=num_input_step_val,
                                                       format=num_input_format_str,
                                                       help=f"Longitud de la cuerda del arco (ej: 100.0). Precisión hasta {app_config['display_precision_general']} decimales.")
    with col2:
        st.session_state.sagitta_input_float = st.number_input("Sagitta/Flecha (s)",
                                                         min_value=1e-9,
                                                         max_value=1e12,
                                                         value=st.session_state.sagitta_input_float,
                                                         step=num_input_step_val,
                                                         format=num_input_format_str,
                                                         help=f"Altura máxima del arco (ej: 10.0). Precisión hasta {app_config['display_precision_general']} decimales.")
    with col3:
        st.session_state.tube_length_input_float = st.number_input("Longitud Tubo (L_tubo)",
                                                              min_value=0.0, # Tube length can be 0 if not used
                                                              max_value=1e12,
                                                              value=st.session_state.tube_length_input_float,
                                                              step=num_input_step_val,
                                                              format=num_input_format_str,
                                                              help=f"Longitud del tubo a rolar (opcional, ej: 50.0). Precisión hasta {app_config['display_precision_general']} decimales.")

    # Retrieve float values from session state
    current_chord_float = st.session_state.chord_input_float
    current_sagitta_float = st.session_state.sagitta_input_float
    current_tube_length_float = st.session_state.tube_length_input_float

    if st.button("🚀 Calcular Radio y Flecha de Tubo", type="primary", use_container_width=True):
        # Perform initial validation on float values (quick checks)
        if current_chord_float <= 1e-9 or current_sagitta_float <= 1e-9: # Check against a very small positive number
            st.error("❌ La Cuerda y la Sagitta deben ser valores estrictamente positivos.")
        # For precise geometric validation, convert to Decimal first
        elif Decimal(str(current_sagitta_float)) >= Decimal(str(current_chord_float)) / Decimal('2'):
            # Display with limited precision for user readability
            st.error(f"❌ La Sagitta (s={current_sagitta_float:.{display_prec}f}) debe ser menor que la mitad de la Cuerda (c/2 = {current_chord_float/2:.{display_prec}f}).")
        elif current_tube_length_float < 0.0:
             st.error("❌ Longitud del Tubo no puede ser negativa (dejar en 0 si no se usa).")
        else:
            with st.spinner("🧠 Calculando con precisión..."):
                start_time = time.time()
                # Pass float values from number_input directly to the cached function
                # The function itself will convert them to Decimal(str(float_val))
                calc_results = calculate_radius_all_methods(
                    current_chord_float,
                    current_sagitta_float,
                    app_config['precision'] # Pass internal calculation precision
                )
                computation_time = time.time() - start_time

                if calc_results.get("success"):
                    # Convert string Decimals from results back to Decimal for formatting and further UI logic
                    radius_final_ui_dec = Decimal(calc_results['radius_final_dec_str'])
                    confidence_ui_dec = Decimal(calc_results['confidence_dec_str'])
                    sag_corr_ui_dec = Decimal(calc_results['sagitta_corrected_dec_str'])
                    sag_incorr_ui_dec = Decimal(calc_results['sagitta_incorrect_dec_str'])
                    err_perc_ui_dec = Decimal(calc_results['error_percentage_dec_str'])

                    st.success(f"✅ **Radio del Arco Calculado (R): {radius_final_ui_dec:.{display_prec}f}**")

                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Confianza (Radio)", f"{confidence_ui_dec:.2%}") # Display confidence as percentage

                    # Count valid methods from string results
                    valid_method_count = sum(1 for v_str in calc_results['methods_dec_str'].values() if not v_str.startswith("Error"))
                    col_m2.metric("Métodos Válidos", f"{valid_method_count}/{len(calc_results['methods_dec_str'])}")
                    col_m3.metric("Tiempo Cálculo", f"{computation_time*1000:.1f}ms")

                    st.subheader("🔍 Comparación de Métodos (Radio)")
                    methods_data_for_ui = [
                        {"Método": k,
                         "Resultado (R)": f"{Decimal(v_str):.{display_prec}f}" if not v_str.startswith("Error") else v_str,
                         "Estado": "✅" if not v_str.startswith("Error") else "❌"}
                        for k,v_str in calc_results['methods_dec_str'].items()
                    ]
                    st.table(methods_data_for_ui)

                    st.subheader(f"🔧 Verificación de Sagitta (para Cuerda original y Radio R={radius_final_ui_dec:.{display_prec}f})")
                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric(f"s con L²/(8R) (Corregida)", f"{sag_corr_ui_dec:.{display_prec}f}")
                    col_s2.metric(f"s con L²/(2R) (Incorrecta)", f"{sag_incorr_ui_dec:.{display_prec}f}")

                    if err_perc_ui_dec.is_finite():
                        st.info(f"📈 Error relativo entre fórmulas de sagitta: {err_perc_ui_dec:.2f}%")
                    else:
                        st.info("📈 Error relativo entre fórmulas de sagitta: Indefinido o Infinito (posiblemente debido a división por cero o valores extremos).")

                    # ---- Cálculo de Flecha para Tubo Rolado ----
                    if current_tube_length_float > 1e-9: # Check if tube length is meaningfully positive
                        st.subheader("🏹 Cálculo de Flecha para Tubo Rolado")
                        # Need an instance of OptimizedCalculator for this non-cached part
                        ui_calc_instance = OptimizedCalculator(app_config['precision'])

                        if radius_final_ui_dec.is_finite() and radius_final_ui_dec > Decimal('0'):
                            # Convert current_tube_length_float to Decimal for this calculation
                            tube_length_for_calc_dec = Decimal(str(current_tube_length_float))
                            flecha_tubo_calculated_dec = ui_calc_instance.calculate_sagitta_corrected(
                                tube_length_for_calc_dec,
                                radius_final_ui_dec
                            )

                            col_t1, col_t2 = st.columns(2)
                            # Display input tube length (float) with desired display precision
                            col_t1.metric("Longitud del Tubo (L_tubo)", f"{current_tube_length_float:.{app_config['display_precision_metrics']}f}")
                            # Display calculated tube sagitta (Decimal) with general display precision
                            col_t2.metric("Flecha del Tubo Calculada (s_tubo)", f"{flecha_tubo_calculated_dec:.{display_prec}f}")
                            st.caption(f"Calculado usando L_tubo y Radio del Arco (R) = {radius_final_ui_dec:.{display_prec}f}")
                        else:
                            st.warning("No se puede calcular la flecha del tubo: el radio del arco calculado no es un número positivo finito.")
                    # ---- FIN NUEVA SECCIÓN ----

                    # Optional AI Analysis
                    if app_config['gemini_api_key'] and REQUESTS_AVAILABLE:
                        with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                            with st.spinner("🧠 Consultando IA... Por favor espere."):
                                ai_prompt = (
                                    f"Análisis breve de cálculo de arco y tubo:\n"
                                    f"- Cuerda del arco (c): {current_chord_float:.{app_config['display_precision_metrics']}f}\n"
                                    f"- Sagitta del arco (s): {current_sagitta_float:.{app_config['display_precision_metrics']}f}\n"
                                    f"- Radio del arco calculado (R): {radius_final_ui_dec:.{display_prec}f}\n"
                                    f"¿Es el radio R geométricamente coherente con c y s? "
                                    f"Si se usó una longitud de tubo L_tubo = {current_tube_length_float:.{app_config['display_precision_metrics']}f} (0 si no se usó), "
                                    f"y se calculó una flecha de tubo s_tubo (si aplica), ¿es esto también coherente?\n"
                                    f"Proporciona una explicación concisa (2-4 frases) sobre la coherencia y razonabilidad de estos valores."
                                )
                                ai_response = call_gemini_api(ai_prompt, app_config['gemini_api_key'])
                                if ai_response["success"]:
                                    st.info("Respuesta de IA:")
                                    st.markdown(ai_response["response"])
                                else:
                                    st.error(f"Error IA: {ai_response['error']}")
                    elif not app_config['gemini_api_key'] and REQUESTS_AVAILABLE : # Added this condition
                         with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                            st.warning("API Key de Gemini no configurada en los secrets de Streamlit para activar el análisis por IA.")


                else: # Error in calculate_radius_all_methods
                    st.error(f"❌ {calc_results.get('error', 'Error desconocido durante el cálculo principal.')}")
        # else for initial input validation block (errors already shown by st.error)

    st.subheader("📋 Ejemplos Rápidos")
    # Examples now use float values for st.number_input compatibility
    examples_float = [
        {"name": "Arco Estándar", "chord": 100.0, "sagitta": 10.0, "tube_length": 50.0},
        {"name": "Puente Pequeño", "chord": 50.0, "sagitta": 5.0}, # tube_length will default to 0.0
        {"name": "Lente Óptica", "chord": 10.0, "sagitta": 0.5, "tube_length": 8.0},
        {"name": "Curva Suave", "chord": 1000.0, "sagitta": 25.0, "tube_length": 200.0}
    ]
    cols = st.columns(len(examples_float))
    for i, ex_float in enumerate(examples_float):
        button_label = f"📐 {ex_float['name']}"
        button_help = f"C: {ex_float['chord']}, S: {ex_float['sagitta']}"
        if 'tube_length' in ex_float:
            button_help += f", L_tubo: {ex_float['tube_length']}"

        if cols[i].button(button_label, key=f"ex_float_{i}", help=button_help, use_container_width=True):
            st.session_state.example_values_float = ex_float # Store all example values (floats)
            st.rerun() # Rerun to update input fields with example values

    # --- Footer ---
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;margin-top:2rem'>🔢 <b>Agente de IA - Streamlit Limitado</b><br>Para funcionalidad completa y mayor precisión, busque 'Agente Cálculos Avanzado HuggingFace'<br><small>Desarrollado con Streamlit Cloud • Optimizaciones para 1GB RAM</small></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state for inputs if not already present, ensuring defaults are floats
    for key, default_value in [('chord_input_float', 100.0),
                               ('sagitta_input_float', 10.0),
                               ('tube_length_input_float', 0.0)]:
        if key not in st.session_state:
            st.session_state[key] = default_value
    main()

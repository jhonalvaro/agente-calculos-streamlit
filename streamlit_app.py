#!/usr/bin/env python3
# """
# 🔢 Calculadora de Precisión para Arcos y Tubos (Streamlit App)
# """

import streamlit as st
import os
import time
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation
import math
import numpy as np

st.set_page_config(
    page_title="🔢 Calculadora Arcos/Tubos",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONTEXT = getcontext()
CONTEXT.prec = 28
CONTEXT.rounding = ROUND_HALF_UP

try:
    import mpmath as mp
    mp.dps = 28
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

@st.cache_resource
def init_app_config():
    return {
        'precision': 28,
        'display_precision_general': 7,
        'display_precision_metrics': 4,
        'max_cache_entries': 20,
        'cache_ttl': 1800,
        'gemini_api_key': st.secrets.get("GOOGLE_API_KEY", "")
    }

class OptimizedCalculator:
    def __init__(self, precision: int = 28):
        self.precision = precision
        if MPMATH_AVAILABLE:
            mp.dps = self.precision

    def calculate_radius_standard(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        if sagitta == Decimal('0'): return Decimal('inf')
        if MPMATH_AVAILABLE:
            c_mp, s_mp = mp.mpf(str(chord)), mp.mpf(str(sagitta))
            if s_mp == 0: return Decimal(str(mp.inf))
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
        return self.calculate_radius_segmental(chord, sagitta)

    def calculate_sagitta_corrected(self, length: Decimal, radius: Decimal) -> Decimal:
        if radius == Decimal('0'): return Decimal('inf')
        if MPMATH_AVAILABLE:
            l_mp, r_mp = mp.mpf(str(length)), mp.mpf(str(radius))
            if r_mp == 0: return Decimal(str(mp.inf))
            return Decimal(str(l_mp**2 / (8 * r_mp)))
        return length**2 / (Decimal('8') * radius)

@st.cache_data(ttl=1800, max_entries=10)
def call_gemini_api(prompt: str, api_key: str) -> Dict[str, Any]:
    if not api_key: return {"success": False, "error": "API key de Gemini no configurada."}
    if not REQUESTS_AVAILABLE: return {"success": False, "error": "Módulo 'requests' no disponible."}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=25)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates: return {"success": False, "error": "Respuesta API: Sin 'candidates'"}
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts: return {"success": False, "error": "Respuesta API: Sin 'parts'"}
        text = parts[0].get("text", "")
        return {"success": True, "response": text}
    except requests.exceptions.Timeout: return {"success": False, "error": "Error API: Timeout (25s)"}
    except requests.exceptions.HTTPError as e: return {"success": False, "error": f"Error API: {e.response.status_code} ({e.response.reason})"}
    except requests.exceptions.RequestException as e: return {"success": False, "error": f"Error de Conexión API: {str(e)}"}
    except Exception as e: return {"success": False, "error": f"Error procesando API: {str(e)}"}

@st.cache_data(ttl=3600, max_entries=20)
def calculate_radius_all_methods(chord_float: float, sagitta_float: float, internal_precision: int) -> Dict[str, Any]:
    calc_context = getcontext()
    calc_context.prec = internal_precision
    try:
        chord = Decimal(str(chord_float))
        sagitta = Decimal(str(sagitta_float))
        if chord <= Decimal('0') or sagitta <= Decimal('0'): return {"error": "Cuerda (c) y Sagitta (s) deben ser > 0."}
        if sagitta >= chord / Decimal('2'): return {"error": f"Sagitta (s={sagitta}) debe ser < mitad de Cuerda (c/2 = {chord/Decimal('2')})."}

        calc_instance = OptimizedCalculator(internal_precision)
        methods_to_run = [("Estándar", calc_instance.calculate_radius_standard),
                          ("Segmental", calc_instance.calculate_radius_segmental),
                          ("Trigonométrico", calc_instance.calculate_radius_trigonometric)]
        results_decimal = {}
        for name, func in methods_to_run:
            try: results_decimal[name] = func(chord, sagitta)
            except InvalidOperation as e: results_decimal[name] = f"Error Decimal: {str(e)}"
            except Exception as e: results_decimal[name] = f"Error ({name}): {str(e)}"

        valid_results_list = [v for v in results_decimal.values() if isinstance(v, Decimal) and v.is_finite() and v > Decimal('0')]
        if not valid_results_list: return {"error": "No se obtuvieron resultados válidos (positivos, finitos)."}

        valid_results_list.sort()
        median_value_dec = valid_results_list[len(valid_results_list) // 2]
        if median_value_dec <= Decimal('0'): return {"error": "Radio mediano calculado no es positivo."}

        confidence_dec = Decimal('1')
        if len(valid_results_list) > 1 and median_value_dec != Decimal('0'):
            max_deviation_dec = max(abs(v - median_value_dec) for v in valid_results_list)
            relative_deviation = min(max_deviation_dec / abs(median_value_dec), Decimal('1'))
            confidence_dec = Decimal('1') - relative_deviation

        sagitta_corrected_dec = calc_instance.calculate_sagitta_corrected(chord, median_value_dec)
        sagitta_incorrect_dec = chord**2 / (Decimal('2') * median_value_dec) if median_value_dec != Decimal('0') else Decimal('inf')
        error_percentage_dec = Decimal('inf')
        if sagitta_corrected_dec.is_finite() and sagitta_corrected_dec != Decimal('0'):
            if sagitta_incorrect_dec.is_finite():
                error_percentage_dec = abs(sagitta_incorrect_dec - sagitta_corrected_dec) / sagitta_corrected_dec * Decimal('100')
        elif sagitta_incorrect_dec == sagitta_corrected_dec: error_percentage_dec = Decimal('0')

        return {"success": True, "radius_final_dec_str": str(median_value_dec),
                "confidence_dec_str": str(confidence_dec),
                "methods_dec_str": {k: str(v) if isinstance(v, Decimal) else v for k, v in results_decimal.items()},
                "sagitta_corrected_dec_str": str(sagitta_corrected_dec),
                "sagitta_incorrect_dec_str": str(sagitta_incorrect_dec),
                "error_percentage_dec_str": str(error_percentage_dec)}
    except InvalidOperation as e: return {"error": f"Error de operación Decimal: {str(e)}"}
    except Exception as e: return {"error": f"Error general en núcleo de cálculo: {str(e)}"}

def generate_arc_plot(chord_dec, sagitta_dec, radius_dec, tube_length_dec=None, tube_sagitta_dec=None, display_precision_cfg=7):
    if not PLOTLY_AVAILABLE:
        return None

    C = float(chord_dec)
    S = float(sagitta_dec)
    R = float(radius_dec)

    if R <= S or R <= C/2 or R == float('inf'):
        fig = go.Figure()
        fig.add_annotation(text="Geometría de arco no válida o infinita para graficar.", xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=200, title_text="Visualización no disponible") # Increased height
        return fig

    h_center = 0.0
    k_center = S - R

    # Clamping value for np.arccos to avoid domain errors due to potential float precision issues
    val_for_arccos = (R - S) / R
    if val_for_arccos > 1.0: val_for_arccos = 1.0
    if val_for_arccos < -1.0: val_for_arccos = -1.0
    alpha = np.arccos(val_for_arccos) # Half angle of the arc segment

    t_angles = np.linspace(-alpha, alpha, 100)
    x_arc = R * np.sin(t_angles)
    y_arc = k_center + R * np.cos(t_angles)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines', name='Arco Calculado', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=[-C/2, C/2], y=[0, 0], mode='lines', name='Cuerda (C)', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, S], mode='lines', name='Sagitta (S)', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=[h_center, 0], y=[k_center, S], mode='lines', name='Radio (R)', line=dict(color='purple', dash='dot')))

    # Annotations
    fig.add_annotation(x=0, y=S/2, text=f"S={S:.{display_precision_cfg}f}", showarrow=False, yshift=10, font=dict(size=10))
    fig.add_annotation(x=C/4, y=0, text=f"C/2={C/2:.{display_precision_cfg}f}", showarrow=False, yshift=-10, font=dict(size=10))
    mid_Rx, mid_Ry = (h_center + 0) / 2, (k_center + S) / 2
    fig.add_annotation(x=mid_Rx, y=mid_Ry, text=f"R={R:.{display_precision_cfg}f}", showarrow=False, bgcolor="rgba(255,255,255,0.7)", font=dict(size=10))


    if tube_length_dec is not None and tube_sagitta_dec is not None and R > 0:
        L_tube = float(tube_length_dec)
        s_tube = float(tube_sagitta_dec)
        # Check if tube can be bent with radius R
        if L_tube / (2 * R) <= 1 and L_tube / (2*R) >= -1 and R > s_tube : # Added R > s_tube check
            alpha_tube = np.arcsin(L_tube / (2 * R))
            k_center_tube_arc = s_tube - R # Center relative to its own chord

            # Offset the tube plot for clarity
            y_offset_for_tube_plot = (k_center - R) - s_tube - (S * 0.3)

            x_arc_tube = R * np.sin(np.linspace(-alpha_tube, alpha_tube, 50))
            y_arc_tube = (k_center_tube_arc + y_offset_for_tube_plot) + R * np.cos(np.linspace(-alpha_tube, alpha_tube, 50))

            y_chord_tube = y_offset_for_tube_plot

            fig.add_trace(go.Scatter(x=x_arc_tube, y=y_arc_tube, mode='lines', name=f'Tubo (L={L_tube:.{display_precision_cfg-2}f}, s={s_tube:.{display_precision_cfg-2}f})', line=dict(color='orange', width=2)))
            fig.add_trace(go.Scatter(x=[-L_tube/2, L_tube/2], y=[y_chord_tube, y_chord_tube], mode='lines', name='Long. Tubo', line=dict(color='brown', dash='dash')))
            fig.add_trace(go.Scatter(x=[0, 0], y=[y_chord_tube, y_chord_tube + s_tube], mode='lines', name='Sag. Tubo', line=dict(color='magenta', dash='dash')))

    fig.update_layout(
        title_text=f"Visualización del Arco (R={R:.{display_precision_cfg}f})",
        xaxis_title="Dimensión X",
        yaxis_title="Dimensión Y",
        yaxis_scaleanchor="x",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(t=50, b=120, l=20, r=20),
        height=600
    )
    return fig

def main():
    app_config = init_app_config()
    getcontext().prec = app_config['precision']
    display_prec_cfg = app_config['display_precision_general']

    st.title("🔢 Calculadora de Precisión para Arcos y Tubos")
    st.info(f"Precisión interna: {app_config['precision']} dígitos. Ingrese datos para calcular radio de arco y/o sagitta de tubo.")

    with st.sidebar:
        st.header("⚙️ Información de la App")
        if MPMATH_AVAILABLE: st.write("✅ mpmath: Disponible")
        else: st.write("⚠️ mpmath: No disponible")
        if PLOTLY_AVAILABLE: st.write("✅ Plotly: Disponible")
        else: st.write("⚠️ Plotly: No disponible")
        if REQUESTS_AVAILABLE:
            if app_config['gemini_api_key']: st.write("✅ API Gemini: Configurada")
            else: st.write("⚠️ API Gemini: Key no configurada")
        else: st.write("❌ API Gemini: No disponible ('requests' faltante)")
        st.write(f"⚙️ Precisión Decimal: {getcontext().prec} dígitos")
        st.markdown("---")
        st.caption("Calculadora geométrica para arcos y tubos.")

    st.header("📐 Entradas para Cálculo")

    for key, default_val in [('chord_input_float', 100.0), ('sagitta_input_float', 10.0), ('tube_length_input_float', 0.0)]:
        if key not in st.session_state: st.session_state[key] = default_val
    if 'example_values_float' in st.session_state:
        st.session_state.chord_input_float = st.session_state.example_values_float['chord']
        st.session_state.sagitta_input_float = st.session_state.example_values_float['sagitta']
        st.session_state.tube_length_input_float = st.session_state.example_values_float.get('tube_length', 0.0)
        del st.session_state.example_values_float

    num_input_fmt_str = f"%.{app_config['display_precision_general']}f"
    num_input_stp_val = 1.0 / (10**app_config['display_precision_general'])

    input_cols_cfg = [0.85, 0.15]

    c1, c2 = st.columns(input_cols_cfg)
    with c1:
        st.session_state.chord_input_float = st.number_input("Cuerda (c)", min_value=1e-9, max_value=1e12,
                                                            value=st.session_state.chord_input_float,
                                                            step=num_input_stp_val, format=num_input_fmt_str,
                                                            key="chord_input_widget",
                                                            help=f"Longitud de la cuerda del arco.")
    with c2:
        st.markdown("##")
        if st.button("✖️", key="clear_chord", help="Limpiar Cuerda"):
            st.session_state.chord_input_float = 0.0 # Default to 0.0 or an empty state if preferred
            st.rerun()

    s1, s2 = st.columns(input_cols_cfg)
    with s1:
        st.session_state.sagitta_input_float = st.number_input("Sagitta/Flecha (s)", min_value=1e-9, max_value=1e12,
                                                                value=st.session_state.sagitta_input_float,
                                                                step=num_input_stp_val, format=num_input_fmt_str,
                                                                key="sagitta_input_widget",
                                                                help=f"Altura máxima del arco.")
    with s2:
        st.markdown("##")
        if st.button("✖️", key="clear_sagitta", help="Limpiar Sagitta"):
            st.session_state.sagitta_input_float = 0.0
            st.rerun()

    lt1, lt2 = st.columns(input_cols_cfg)
    with lt1:
        st.session_state.tube_length_input_float = st.number_input("Longitud Tubo (L_tubo)", min_value=0.0, max_value=1e12,
                                                                    value=st.session_state.tube_length_input_float,
                                                                    step=num_input_stp_val, format=num_input_fmt_str,
                                                                    key="tube_length_input_widget",
                                                                    help=f"Longitud del tubo a rolar (opcional).")
    with lt2:
        st.markdown("##")
        if st.button("✖️", key="clear_tube_length", help="Limpiar Longitud Tubo"):
            st.session_state.tube_length_input_float = 0.0
            st.rerun()


    if st.button("🚀 Calcular", type="primary", use_container_width=True):
        current_chord_f = st.session_state.chord_input_float
        current_sagitta_f = st.session_state.sagitta_input_float
        current_tube_len_f = st.session_state.tube_length_input_float

        if current_chord_f <= 1e-9 or current_sagitta_f <= 1e-9: st.error("❌ Cuerda y Sagitta deben ser > 0.")
        elif Decimal(str(current_sagitta_f)) >= Decimal(str(current_chord_f)) / Decimal('2'): st.error(f"❌ Sagitta (s={current_sagitta_f:.{display_prec_cfg}f}) debe ser < mitad de Cuerda (c/2 = {current_chord_f/2:.{display_prec_cfg}f}).")
        elif current_tube_len_f < 0.0: st.error("❌ Longitud del Tubo no puede ser negativa.")
        else:
            with st.spinner("🧠 Calculando..."):
                start_time = time.time()
                calc_results = calculate_radius_all_methods(current_chord_f, current_sagitta_f, app_config['precision'])
                computation_time = time.time() - start_time

                if calc_results.get("success"):
                    radius_final_dec = Decimal(calc_results['radius_final_dec_str'])
                    confidence_dec = Decimal(calc_results['confidence_dec_str'])
                    st.success(f"✅ **Radio del Arco Calculado (R): {radius_final_dec:.{display_prec_cfg}f}**")

                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Confianza (Radio)", f"{confidence_dec:.2%}")
                    valid_method_cnt = sum(1 for v in calc_results['methods_dec_str'].values() if not v.startswith("Error"))
                    col_m2.metric("Métodos Válidos", f"{valid_method_cnt}/{len(calc_results['methods_dec_str'])}")
                    col_m3.metric("Tiempo Cálculo", f"{computation_time*1000:.1f}ms")

                    st.subheader("Detalles del Cálculo del Radio")
                    methods_ui_data = [{"Método": k, "Resultado (R)": f"{Decimal(v):.{display_prec_cfg}f}" if not v.startswith("Error") else v, "Estado": "✅" if not v.startswith("Error") else "❌"} for k,v in calc_results['methods_dec_str'].items()]
                    st.table(methods_ui_data)

                    sag_corr_dec = Decimal(calc_results['sagitta_corrected_dec_str'])
                    sag_incorr_dec = Decimal(calc_results['sagitta_incorrect_dec_str'])
                    err_perc_dec = Decimal(calc_results['error_percentage_dec_str'])
                    st.subheader(f"Verificación de Sagitta del Arco")
                    col_s1,col_s2=st.columns(2); col_s1.metric(f"s con L²/(8R)",f"{sag_corr_dec:.{display_prec_cfg}f}"); col_s2.metric(f"s con L²/(2R)",f"{sag_incorr_dec:.{display_prec_cfg}f}")
                    if err_perc_dec.is_finite(): st.info(f"Error relativo (sagitta): {err_perc_dec:.2f}%")
                    else: st.info("Error relativo (sagitta): Indefinido.")

                    flecha_tubo_calc_dec = None
                    if current_tube_len_f > 1e-9:
                        st.subheader("🏹 Cálculo de Flecha para Tubo")
                        ui_calc = OptimizedCalculator(app_config['precision'])
                        if radius_final_dec.is_finite() and radius_final_dec > Decimal('0'):
                            tube_len_dec = Decimal(str(current_tube_len_f))
                            flecha_tubo_calc_dec = ui_calc.calculate_sagitta_corrected(tube_len_dec, radius_final_dec)
                            col_t1,col_t2=st.columns(2); col_t1.metric("Longitud Tubo",f"{current_tube_len_f:.{app_config['display_precision_metrics']}f}"); col_t2.metric("Flecha Tubo Calculada",f"{flecha_tubo_calc_dec:.{display_prec_cfg}f}")
                            st.caption(f"Para L_tubo={current_tube_len_f:.{app_config['display_precision_metrics']}f}, R_arco={radius_final_dec:.{display_prec_cfg}f}")
                        else: st.warning("No se calcula flecha de tubo: radio del arco no válido.")

                    st.subheader("📊 Visualización del Arco")
                    plot_fig_obj = generate_arc_plot(
                        Decimal(str(current_chord_f)), Decimal(str(current_sagitta_f)), radius_final_dec,
                        Decimal(str(current_tube_len_f)) if current_tube_len_f > 1e-9 else None,
                        flecha_tubo_calc_dec,
                        display_prec_cfg
                    )
                    if plot_fig_obj and PLOTLY_AVAILABLE: st.plotly_chart(plot_fig_obj, use_container_width=True)
                    elif PLOTLY_AVAILABLE: st.warning("No se pudo generar gráfico.")
                    else: st.info("Gráficos desactivados (Plotly no disponible).")

                    if app_config['gemini_api_key'] and REQUESTS_AVAILABLE:
                        with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                            with st.spinner("🧠 Consultando IA..."):
                                s_tubo_str = f"{flecha_tubo_calc_dec:.{display_prec_cfg}f}" if flecha_tubo_calc_dec is not None else "N/A"
                                ai_prompt = (f"Análisis de cálculo de arco:\n"
                                             f"Cuerda (c): {current_chord_f:.{app_config['display_precision_metrics']}f}, Sagitta (s): {current_sagitta_f:.{app_config['display_precision_metrics']}f}, Radio calculado (R): {radius_final_dec:.{display_prec_cfg}f}.\n"
                                             f"¿Es R coherente con c y s? \n"
                                             f"Si L_tubo = {current_tube_len_f:.{app_config['display_precision_metrics']}f} (>0), y s_tubo = {s_tubo_str}, ¿es s_tubo coherente?\n"
                                             f"Explicación concisa.")
                                ai_response = call_gemini_api(ai_prompt, app_config['gemini_api_key'])
                                if ai_response["success"]: st.info("Respuesta de IA:"); st.markdown(ai_response["response"])
                                else: st.error(f"Error IA: {ai_response['error']}")
                    elif not app_config['gemini_api_key'] and REQUESTS_AVAILABLE:
                         with st.expander("🤖 Análisis con IA (Opcional)", expanded=False): st.warning("API Key de Gemini no configurada.")
                else:
                    st.error(f"❌ {calc_results.get('error', 'Error en cálculo principal.')}")

    st.subheader("📋 Ejemplos")
    examples_float_data = [
        {"name": "Arco Estándar", "chord": 100.0, "sagitta": 10.0, "tube_length": 50.0},
        {"name": "Puente Pequeño", "chord": 50.0, "sagitta": 5.0},
        {"name": "Lente Óptica", "chord": 10.0, "sagitta": 0.5, "tube_length": 8.0},
        {"name": "Curva Suave", "chord": 1000.0, "sagitta": 25.0, "tube_length": 200.0}]
    cols_ex = st.columns(len(examples_float_data))
    for idx, ex_data in enumerate(examples_float_data):
        ex_btn_label = f"📐 {ex_data['name']}"
        ex_btn_help = f"C: {ex_data['chord']}, S: {ex_data['sagitta']}"
        if 'tube_length' in ex_data: ex_btn_help += f", L_tubo: {ex_data['tube_length']}"
        if cols_ex[idx].button(ex_btn_label, key=f"ex_btn_{idx}", help=ex_btn_help, use_container_width=True):
            st.session_state.example_values_float = ex_data
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;margin-top:1rem'><small>Calculadora de Precisión Arcos/Tubos</small></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    for key, default_val in [('chord_input_float', 100.0), ('sagitta_input_float', 10.0), ('tube_length_input_float', 0.0)]:
        if key not in st.session_state: st.session_state[key] = default_val
    main()

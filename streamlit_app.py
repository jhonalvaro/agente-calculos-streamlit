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

UNITS_TO_METERS = {
    "Metros (m)": Decimal("1.0"),
    "Centímetros (cm)": Decimal("0.01"),
    "Milímetros (mm)": Decimal("0.001"),
    "Pulgadas (in)": Decimal("0.0254"),
    "Pies (ft)": Decimal("0.3048")
}
UNIT_NAMES = list(UNITS_TO_METERS.keys())

try:
    import mpmath as mp; mp.dps = 28; MPMATH_AVAILABLE = True
except ImportError: MPMATH_AVAILABLE = False
try:
    import requests; REQUESTS_AVAILABLE = True
except ImportError: REQUESTS_AVAILABLE = False
try:
    import plotly.graph_objects as go; PLOTLY_AVAILABLE = True
except ImportError: PLOTLY_AVAILABLE = False

@st.cache_resource
def init_app_config():
    return {'precision': 28, 'display_precision_general': 1, 'display_precision_metrics': 1,
            'max_cache_entries': 20, 'cache_ttl': 1800, 'gemini_api_key': st.secrets.get("GOOGLE_API_KEY", "")}

class OptimizedCalculator:
    def __init__(self, precision: int = 28):
        self.precision = precision
        if MPMATH_AVAILABLE: mp.dps = self.precision

    def _calc_radius(self, chord: Decimal, sagitta: Decimal) -> Decimal: # Helper for common formula part
        return sagitta/Decimal('2') + chord**2/(Decimal('8')*sagitta)

    def calculate_radius_standard(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        if sagitta == Decimal('0'): return Decimal('inf')
        # (c^2 + 4s^2) / (8s) = c^2/(8s) + 4s^2/(8s) = c^2/(8s) + s/2
        return self._calc_radius(chord, sagitta)

    def calculate_radius_segmental(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        if sagitta == Decimal('0'): return Decimal('inf')
        return self._calc_radius(chord, sagitta)

    def calculate_radius_trigonometric(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        return self._calc_radius(chord, sagitta)

    def calculate_sagitta_corrected(self, length: Decimal, radius: Decimal) -> Decimal:
        if radius == Decimal('0'): return Decimal('inf')
        return length**2 / (Decimal('8') * radius)

    def calculate_arc_length_and_angle(self, chord: Decimal, sagitta: Decimal, radius: Decimal) -> Dict[str, Optional[Decimal]]:
        current_prec = getcontext().prec
        # Basic geometric checks
        if not (radius > Decimal('0') and not radius.is_infinite() and not radius.is_nan()): return {"arc_length": None, "central_angle_deg": None, "error": "Radio no válido."}
        if not (sagitta > Decimal('0') and sagitta < radius * 2): return {"arc_length": None, "central_angle_deg": None, "error": "Sagitta no válida."} # s < diameter
        if not (chord > Decimal('0')): return {"arc_length": None, "central_angle_deg": None, "error": "Cuerda no válida."}
        if radius < sagitta : return {"arc_length": None, "central_angle_deg": None, "error": "Radio debe ser >= Sagitta."} # R=S means semicircle
        # For a circular segment, R must be >= C/2. If R = C/2, it's a semicircle.
        # If R < C/2, the chord doesn't fit in the circle.
        # This check is implicitly handled by acos domain if R-S is used.
        # (R-S)/R must be in [-1,1]. R-S is distance from center to chord.
        # If R-S > R (S<0) or R-S < -R (S>2R, impossible if S<D)

        h = radius - sagitta # Distance from center to chord midpoint
        # Ensure argument for acos is within [-1, 1]
        val_for_acos_dec = h / radius
        if val_for_acos_dec > Decimal('1'): val_for_acos_dec = Decimal('1')
        elif val_for_acos_dec < Decimal('-1'): val_for_acos_dec = Decimal('-1')

        try:
            if MPMATH_AVAILABLE:
                mp.dps = current_prec
                radius_mp = mp.mpf(str(radius)) # Only radius needed for mpmath part if using h/R
                val_for_acos_mp = mp.mpf(str(val_for_acos_dec))

                alpha_rad_mp = mp.acos(val_for_acos_mp)
                theta_rad_mp = 2 * alpha_rad_mp
                arc_length_mp = radius_mp * theta_rad_mp
                central_angle_deg_mp = mp.degrees(theta_rad_mp)
                return {"arc_length": Decimal(str(arc_length_mp)), "central_angle_deg": Decimal(str(central_angle_deg_mp)), "error": None}
            else:
                val_for_acos_float = float(val_for_acos_dec)
                alpha_rad_float = math.acos(val_for_acos_float)
                theta_rad_float = 2 * alpha_rad_float
                arc_length_dec = radius * Decimal(str(theta_rad_float))
                central_angle_deg_dec = Decimal(str(math.degrees(theta_rad_float)))
                return {"arc_length": arc_length_dec, "central_angle_deg": central_angle_deg_dec, "error": None}
        except ValueError as ve: return {"arc_length": None, "central_angle_deg": None, "error": f"Error de valor en cálculo de ángulo: {ve}"}
        except Exception as e: return {"arc_length": None, "central_angle_deg": None, "error": f"Error inesperado en cálculo de arco: {e}"}

@st.cache_data(ttl=1800, max_entries=10)
def call_gemini_api(prompt: str, api_key: str) -> Dict[str, Any]: # ... (no changes) ...
    if not api_key: return {"success": False, "error": "API key de Gemini no configurada."}
    if not REQUESTS_AVAILABLE: return {"success": False, "error": "Módulo 'requests' no disponible."}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=25); response.raise_for_status()
        data = response.json(); candidates = data.get("candidates", [])
        if not candidates: return {"success": False, "error": "Respuesta API: Sin 'candidates'"}
        content = candidates[0].get("content", {}); parts = content.get("parts", [])
        if not parts: return {"success": False, "error": "Respuesta API: Sin 'parts'"}
        text = parts[0].get("text", ""); return {"success": True, "response": text}
    except requests.exceptions.Timeout: return {"success": False, "error": "Error API: Timeout (25s)"}
    except requests.exceptions.HTTPError as e: return {"success": False, "error": f"Error API: {e.response.status_code} ({e.response.reason})"}
    except requests.exceptions.RequestException as e: return {"success": False, "error": f"Error de Conexión API: {str(e)}"}
    except Exception as e: return {"success": False, "error": f"Error procesando API: {str(e)}"}


@st.cache_data(ttl=3600, max_entries=20)
def calculate_radius_all_methods(chord_base_unit_float: float, sagitta_base_unit_float: float, internal_precision: int) -> Dict[str, Any]:
    # Inputs (chord_base_unit_float, sagitta_base_unit_float) are now expected to be in the base unit (meters)
    calc_context = getcontext(); calc_context.prec = internal_precision
    try:
        chord_base = Decimal(str(chord_base_unit_float))
        sagitta_base = Decimal(str(sagitta_base_unit_float))
        arc_length_str, central_angle_str, arc_calc_error_msg = "N/A", "N/A", None

        if chord_base <= Decimal('0') or sagitta_base <= Decimal('0'): return {"error": "Cuerda y Sagitta (en unidad base) deben ser > 0."}
        if sagitta_base >= chord_base / Decimal('2'): return {"error": f"Sagitta ({sagitta_base}) debe ser < mitad de Cuerda ({chord_base/Decimal('2')}) en unidad base."}

        calc_instance = OptimizedCalculator(internal_precision)
        methods_to_run = [("Estándar", calc_instance.calculate_radius_standard),
                          ("Segmental", calc_instance.calculate_radius_segmental),
                          ("Trigonométrico", calc_instance.calculate_radius_trigonometric)]
        results_decimal = {}
        for name, func in methods_to_run:
            try: results_decimal[name] = func(chord_base, sagitta_base) # Use base unit values
            except Exception as e: results_decimal[name] = f"Error ({name}): {str(e)}"

        valid_results_list = [v for v in results_decimal.values() if isinstance(v, Decimal) and v.is_finite() and v > Decimal('0')]
        if not valid_results_list: return {"error": "No se obtuvieron resultados válidos para el radio (en unidad base)."}
        valid_results_list.sort(); median_radius_base = valid_results_list[len(valid_results_list) // 2]
        if median_radius_base <= Decimal('0'): return {"error": "Radio mediano (unidad base) no es positivo."}

        arc_length_data = calc_instance.calculate_arc_length_and_angle(chord_base, sagitta_base, median_radius_base)
        if arc_length_data.get("error"): arc_calc_error_msg = arc_length_data["error"]
        else:
            arc_len_val, central_angle_val = arc_length_data.get("arc_length"), arc_length_data.get("central_angle_deg")
            if arc_len_val is not None: arc_length_str = str(arc_len_val)
            if central_angle_val is not None: central_angle_str = str(central_angle_val)

        confidence_dec = Decimal('1')
        if len(valid_results_list) > 1: # median_radius_base is already checked > 0
            max_dev = max(abs(v - median_radius_base) for v in valid_results_list)
            confidence_dec = Decimal('1') - min(max_dev / median_radius_base, Decimal('1'))

        sag_corr_base = calc_instance.calculate_sagitta_corrected(chord_base, median_radius_base)
        sag_incorr_base = chord_base**2 / (Decimal('2')*median_radius_base) if median_radius_base!=Decimal('0') else Decimal('inf')
        err_perc_dec = Decimal('inf')
        if sag_corr_base.is_finite() and sag_corr_base != Decimal('0'):
            if sag_incorr_base.is_finite(): err_perc_dec = abs(sag_incorr_base - sag_corr_base) / sag_corr_base * Decimal('100')
        elif sag_incorr_base == sag_corr_base: err_perc_dec = Decimal('0')

        # All returned dimensional strings are in base units (meters)
        return {"success": True, "radius_final_dec_str": str(median_radius_base),
                "confidence_dec_str": str(confidence_dec),
                "methods_dec_str": {k: str(v) if isinstance(v,Decimal) else v for k,v in results_decimal.items()},
                "sagitta_corrected_dec_str": str(sag_corr_base),
                "sagitta_incorrect_dec_str": str(sag_incorr_base),
                "error_percentage_dec_str": str(err_perc_dec),
                "arc_length_dec_str": arc_length_str,
                "central_angle_deg_str": central_angle_str,
                "arc_calculation_error": arc_calc_error_msg }
    except Exception as e: return {"error": f"Error general en núcleo de cálculo: {str(e)}"}


def create_single_arc_visualization(chord_val, sagitta_val, radius_val, plot_title_prefix="Arco", display_precision_cfg=1, unit_name="Unidades"):
    # Expects chord_val, sagitta_val, radius_val to be ALREADY in the user's selected display unit
    if not PLOTLY_AVAILABLE: return None
    C, S, R = float(chord_val), float(sagitta_val), float(radius_val) # Values are for display

    # Validation based on display values
    if not (R > 1e-9 and S > 1e-9 and C > 1e-9 and R != float('inf') and S < C/2 and R >= S-(1e-9) and R >= C/2-(1e-9)): # Added tolerance for R>=S, R>=C/2
        fig = go.Figure()
        fig.add_annotation(text=f"Geometría '{plot_title_prefix}' no válida para graficar (C={C:.{display_precision_cfg}f}, S={S:.{display_precision_cfg}f}, R={R:.{display_precision_cfg}f} {unit_name}).", xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=200, title_text=f"{plot_title_prefix}: Visualización no disponible")
        return fig

    h_center, k_center = 0.0, S - R
    val_for_arccos = np.clip((R - S) / R, -1.0, 1.0) # Clip to handle potential float inaccuracies
    alpha = np.arccos(val_for_arccos)
    t_angles = np.linspace(-alpha, alpha, 100)
    x_arc, y_arc = R * np.sin(t_angles), k_center + R * np.cos(t_angles)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines', name='Arco', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=[-C/2, C/2], y=[0, 0], mode='lines', name='Cuerda', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, S], mode='lines', name='Sagitta', line=dict(color='green', dash='dash')))
    fig.add_annotation(x=0, y=S * 0.5, text=f"S={S:.{display_precision_cfg}f}", showarrow=False, yshift=10, font=dict(size=10))
    fig.add_annotation(x=0, y=-S*0.1, text=f"C={C:.{display_precision_cfg}f}", showarrow=False, yshift=-5, font=dict(size=10))
    fig.update_layout(title_text=f"{plot_title_prefix} (Radio R={R:.{display_precision_cfg}f} {unit_name})",
                      xaxis_title=f"Dimensión X ({unit_name})", yaxis_title=f"Dimensión Y ({unit_name})",
                      yaxis_scaleanchor="x", legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                      margin=dict(t=50, b=120, l=20, r=20), height=500)
    return fig

def main():
    app_config = init_app_config()
    getcontext().prec = app_config['precision']
    display_prec_cfg = app_config['display_precision_general']

    st.title("🔢 Calculadora de Precisión para Arcos y Tubos")

    if 'selected_unit_name' not in st.session_state: st.session_state.selected_unit_name = UNIT_NAMES[0]
    newly_selected_unit_name = st.selectbox("Unidad para Entradas/Resultados:", UNIT_NAMES,
                                            index=UNIT_NAMES.index(st.session_state.selected_unit_name),
                                            key="unit_selector_widget")
    if newly_selected_unit_name != st.session_state.selected_unit_name:
        st.session_state.selected_unit_name = newly_selected_unit_name; st.rerun()

    selected_unit_name_for_display = st.session_state.selected_unit_name
    factor_to_base_unit = UNITS_TO_METERS[selected_unit_name_for_display]

    st.info(f"Precisión interna: {app_config['precision']} dígitos. Entradas/Resultados en {selected_unit_name_for_display}. Resultados mostrados con {display_prec_cfg} decimal.")

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
        st.markdown("---"); st.caption("Calculadora geométrica para arcos y tubos.")

    st.header("📐 Entradas para Cálculo")

    for key, default_val in [('chord_input_float', 100.0), ('sagitta_input_float', 10.0), ('tube_length_input_float', 0.0)]:
        if key not in st.session_state: st.session_state[key] = default_val
    if 'example_values_float' in st.session_state:
        st.session_state.chord_input_float = st.session_state.example_values_float['chord']
        st.session_state.sagitta_input_float = st.session_state.example_values_float['sagitta']
        st.session_state.tube_length_input_float = st.session_state.example_values_float.get('tube_length', 0.0)
        st.session_state.selected_unit_name = UNIT_NAMES[0]; del st.session_state.example_values_float; st.rerun()

    num_input_fmt_str = f"%.{display_prec_cfg}f"
    num_input_stp_val = 1.0 / (10**display_prec_cfg)

    st.session_state.chord_input_float = st.number_input(f"Cuerda (c) en {selected_unit_name_for_display}", min_value=1e-9, max_value=1e12, value=st.session_state.chord_input_float, step=num_input_stp_val, format=num_input_fmt_str, key="chord_input_widget", help=f"Longitud de la cuerda del arco.")
    st.session_state.sagitta_input_float = st.number_input(f"Sagitta/Flecha (s) en {selected_unit_name_for_display}", min_value=1e-9, max_value=1e12, value=st.session_state.sagitta_input_float, step=num_input_stp_val, format=num_input_fmt_str, key="sagitta_input_widget", help=f"Altura máxima del arco.")
    st.session_state.tube_length_input_float = st.number_input(f"Longitud Tubo (L_tubo) en {selected_unit_name_for_display}", min_value=0.0, max_value=1e12, value=st.session_state.tube_length_input_float, step=num_input_stp_val, format=num_input_fmt_str, key="tube_length_input_widget", help=f"Longitud del tubo a rolar (opcional).")

    if st.button("🚀 Calcular", type="primary", use_container_width=True):
        current_chord_f_selected_unit = st.session_state.chord_input_float
        current_sagitta_f_selected_unit = st.session_state.sagitta_input_float
        current_tube_len_f_selected_unit = st.session_state.tube_length_input_float

        chord_base_f = current_chord_f_selected_unit * float(factor_to_base_unit)
        sagitta_base_f = current_sagitta_f_selected_unit * float(factor_to_base_unit)

        if current_chord_f_selected_unit <= 1e-9 or current_sagitta_f_selected_unit <= 1e-9: st.error("❌ Cuerda y Sagitta deben ser > 0.")
        elif Decimal(str(current_sagitta_f_selected_unit)) >= Decimal(str(current_chord_f_selected_unit)) / Decimal('2'):
            st.error(f"❌ Sagitta (s={current_sagitta_f_selected_unit:.{display_prec_cfg}f}) debe ser < mitad de Cuerda (c/2 = {current_chord_f_selected_unit/2:.{display_prec_cfg}f}).")
        elif current_tube_len_f_selected_unit < 0.0: st.error("❌ Longitud del Tubo no puede ser negativa.")
        else:
            with st.spinner("🧠 Calculando..."):
                start_time = time.time()
                calc_results = calculate_radius_all_methods(chord_base_f, sagitta_base_f, app_config['precision'])
                computation_time = time.time() - start_time

                if calc_results.get("success"):
                    radius_base_dec = Decimal(calc_results['radius_final_dec_str'])
                    arc_length_base_str = calc_results.get('arc_length_dec_str', "N/A")
                    central_angle_base_str = calc_results.get('central_angle_deg_str', "N/A") # Angle is unitless
                    arc_calc_err = calc_results.get('arc_calculation_error')

                    radius_display = radius_base_dec / factor_to_base_unit

                    arc_length_display_val, central_angle_display_val = None, None
                    if arc_length_base_str != "N/A" and not arc_calc_err:
                        try: arc_length_display_val = Decimal(arc_length_base_str) / factor_to_base_unit
                        except: arc_calc_err = arc_calc_err or "Error al convertir long. arco para display"
                    if central_angle_base_str != "N/A": # Angle doesn't need unit conversion for display
                        try: central_angle_display_val = Decimal(central_angle_base_str)
                        except: arc_calc_err = arc_calc_err or "Error al convertir ángulo para display"


                    st.success(f"✅ **Radio del Arco Principal (R): {radius_display:.{display_prec_cfg}f} {selected_unit_name_for_display}**")

                    if arc_calc_err: st.warning(f"Info cálculo de arco: {arc_calc_err}")
                    elif arc_length_display_val is not None and arc_length_display_val > Decimal('1e-7'):
                        st.subheader("📏 Dimensiones del Arco Principal")
                        col_L, col_A = st.columns(2)
                        col_L.metric("Longitud del Arco (L_arco)", f"{arc_length_display_val:.{display_prec_cfg}f} {selected_unit_name_for_display}")
                        if central_angle_display_val is not None:
                             col_A.metric("Ángulo Central", f"{central_angle_display_val:.{display_prec_cfg}f}°")

                    confidence_dec = Decimal(calc_results['confidence_dec_str']) # Unitless
                    col_m1,col_m2,col_m3=st.columns(3); col_m1.metric("Confianza (Radio)",f"{confidence_dec:.2%}")
                    valid_m_cnt=sum(1 for v in calc_results['methods_dec_str'].values() if not v.startswith("Error"))
                    col_m2.metric("Métodos Válidos",f"{valid_m_cnt}/{len(calc_results['methods_dec_str'])}"); col_m3.metric("Tiempo Cálculo",f"{computation_time*1000:.1f}ms")

                    st.subheader(f"Detalles del Cálculo del Radio ({selected_unit_name_for_display})")
                    methods_ui_data = [{"Método": k, f"Resultado (R) en {selected_unit_name_for_display}": f"{Decimal(v) / factor_to_base_unit:.{display_prec_cfg}f}" if not v.startswith("Error") else v, "Estado": "✅" if not v.startswith("Error") else "❌"} for k,v in calc_results['methods_dec_str'].items()]
                    st.table(methods_ui_data)

                    sag_corr_base_dec = Decimal(calc_results['sagitta_corrected_dec_str'])
                    sag_incorr_base_dec = Decimal(calc_results['sagitta_incorrect_dec_str'])
                    err_perc_dec = Decimal(calc_results['error_percentage_dec_str']) # Unitless
                    sag_corr_display = sag_corr_base_dec / factor_to_base_unit
                    sag_incorr_display = sag_incorr_base_dec / factor_to_base_unit if sag_incorr_base_dec.is_finite() else Decimal('inf')

                    st.subheader(f"Verificación de Sagitta (Arco Principal)")
                    col_s1,col_s2=st.columns(2); col_s1.metric(f"s con L²/(8R)",f"{sag_corr_display:.{display_prec_cfg}f} {selected_unit_name_for_display}"); col_s2.metric(f"s con L²/(2R)",f"{sag_incorr_display:.{display_prec_cfg}f} {selected_unit_name_for_display}")
                    if err_perc_dec.is_finite(): st.info(f"Error relativo (sagitta): {err_perc_dec:.1f}%")
                    else: st.info("Error relativo (sagitta): Indefinido.")

                    st.subheader("📊 Visualización del Arco Principal")
                    main_arc_plot_fig = create_single_arc_visualization(Decimal(str(current_chord_f_selected_unit)), Decimal(str(current_sagitta_f_selected_unit)), radius_display, plot_title_prefix="Arco Principal", display_precision_cfg=display_prec_cfg, unit_name=selected_unit_name_for_display)
                    if main_arc_plot_fig and PLOTLY_AVAILABLE: st.plotly_chart(main_arc_plot_fig, use_container_width=True)

                    flecha_tubo_calc_display = None
                    num_tubes_display_str = "N/A"

                    if current_tube_len_f_selected_unit > 1e-9:
                        st.subheader("🏹 Cálculo de Flecha para Tubo")
                        ui_calc = OptimizedCalculator(app_config['precision'])
                        if radius_base_dec.is_finite() and radius_base_dec > Decimal('0'):
                            tube_len_base_dec = Decimal(str(current_tube_len_f_selected_unit)) * factor_to_base_unit
                            flecha_tubo_base_dec = ui_calc.calculate_sagitta_corrected(tube_len_base_dec, radius_base_dec)
                            flecha_tubo_calc_display = flecha_tubo_base_dec / factor_to_base_unit

                            col_t1,col_t2=st.columns(2); col_t1.metric(f"Longitud Tubo ({selected_unit_name_for_display})",f"{current_tube_len_f_selected_unit:.{app_config['display_precision_metrics']}f}"); col_t2.metric(f"Flecha Tubo Calculada ({selected_unit_name_for_display})",f"{flecha_tubo_calc_display:.{display_prec_cfg}f}")
                            st.caption(f"Para L_tubo y R_arco en {selected_unit_name_for_display}")

                            st.subheader("📊 Visualización del Tubo Rolado")
                            # Pass tube length in selected units, calculated tube sagitta in selected units, and radius in selected units
                            tube_arc_plot_fig = create_single_arc_visualization(Decimal(str(current_tube_len_f_selected_unit)), flecha_tubo_calc_display, radius_display, plot_title_prefix="Tubo Rolado", display_precision_cfg=display_prec_cfg, unit_name=selected_unit_name_for_display)
                            if tube_arc_plot_fig and PLOTLY_AVAILABLE: st.plotly_chart(tube_arc_plot_fig, use_container_width=True)

                            if arc_length_display_val is not None and arc_length_display_val > Decimal('1e-7') and current_tube_len_f_selected_unit > 1e-9 and not arc_calc_err:
                                num_tubes_float = float(arc_length_display_val) / current_tube_len_f_selected_unit
                                full_tubes = math.floor(num_tubes_float)
                                num_tubes_display_str = f"{num_tubes_float:.{display_prec_cfg}f} ({full_tubes} completo(s))"
                                st.subheader("🧩 Ajuste de Tubos en el Arco")
                                st.write(f"L_arco: **{arc_length_display_val:.{display_prec_cfg}f} {selected_unit_name_for_display}** | L_tubo: **{current_tube_len_f_selected_unit:.{app_config['display_precision_metrics']}f} {selected_unit_name_for_display}**")
                                st.metric(label="Número de Tubos que Caben", value=f"{num_tubes_float:.{display_prec_cfg}f}")
                                if full_tubes > 0: st.caption(f"Esto equivale a **{full_tubes}** tubo(s) completo(s).")
                            elif arc_calc_err: st.caption(f"Ajuste de tubos no calculado: {arc_calc_err}")
                        else: st.warning("No se calcula flecha de tubo: radio del arco principal no válido.")

                    if app_config['gemini_api_key'] and REQUESTS_AVAILABLE:
                        with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                            with st.spinner("🧠 Consultando IA..."):
                                s_tubo_str_ai = f"{flecha_tubo_calc_display:.{display_prec_cfg}f} {selected_unit_name_for_display}" if flecha_tubo_calc_display is not None else "N/A"
                                arc_len_str_ai = f"{arc_length_display_val:.{display_prec_cfg}f} {selected_unit_name_for_display}" if arc_length_display_val is not None and arc_length_display_val > Decimal('1e-7') and not arc_calc_err else "N/A"
                                central_angle_str_ai = f"{central_angle_display_val:.{display_prec_cfg}f}°" if central_angle_display_val is not None and not arc_calc_err else "N/A"

                                prompt_lines = []
                                prompt_lines.append(f"Análisis de cálculo de arco (unidades de entrada/salida: {selected_unit_name_for_display}):")
                                prompt_lines.append(f"- Cuerda (c): {current_chord_f_selected_unit:.{app_config['display_precision_metrics']}f}, Sagitta (s): {current_sagitta_f_selected_unit:.{app_config['display_precision_metrics']}f}, Radio calculado (R): {radius_display:.{display_prec_cfg}f}.")
                                prompt_lines.append(f"- Longitud de arco calculada (L_arco): {arc_len_str_ai}, Ángulo Central: {central_angle_str_ai}.")
                                prompt_lines.append(f"¿Es R geométricamente coherente con c y s? ¿Son L_arco y Ángulo coherentes?")
                                prompt_lines.append(f"Si L_tubo = {current_tube_len_f_selected_unit:.{app_config['display_precision_metrics']}f} (>0), y s_tubo = {s_tubo_str_ai}, ¿es s_tubo coherente?")
                                prompt_lines.append(f"Si L_arco > 0 y L_tubo > 0, ¿cuántos tubos caben ({num_tubes_display_str})?")
                                prompt_lines.append(f"Explicación concisa.")
                                ai_prompt = "\n".join(prompt_lines)

                                ai_response = call_gemini_api(ai_prompt, app_config['gemini_api_key'])
                                if ai_response["success"]: st.info("Respuesta de IA:"); st.markdown(ai_response["response"])
                                else: st.error(f"Error IA: {ai_response['error']}")
                    # ... (rest of AI expander and main function)
                else:
                    st.error(f"❌ {calc_results.get('error', 'Error en cálculo principal.')}")

    # ... (Examples and JS script remain the same) ...
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
            st.session_state.selected_unit_name = UNIT_NAMES[0] # Reset to meters for examples
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;margin-top:1rem'><small>Calculadora de Precisión Arcos/Tubos</small></div>", unsafe_allow_html=True)
    js_script = ('<script>\n'
                 'try {\n'
                 '    const numberInputs = document.querySelectorAll(\'input[type="number"]\');\n'
                 '    numberInputs.forEach(function(input) {\n'
                 '        input.addEventListener(\'focus\', function() { this.select(); });\n'
                 '    });\n'
                 '} catch (e) { /* console.error(\'Error attaching focus listeners:\', e); */ }\n'
                 '</script>')
    st.markdown(js_script, unsafe_allow_html=True)

if __name__ == "__main__":
    for key, default_val in [('chord_input_float', 100.0), ('sagitta_input_float', 10.0), ('tube_length_input_float', 0.0), ('selected_unit_name', UNIT_NAMES[0])]:
        if key not in st.session_state: st.session_state[key] = default_val
    main()

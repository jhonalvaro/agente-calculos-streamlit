# File aggressively overwritten on 2024-07-15 to ensure removal of all stray markdown syntax.
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

# Precisión de decimales para mostrar según la unidad
UNIT_DECIMAL_PRECISION = {
    "Metros (m)": 2,        # 0.01
    "Centímetros (cm)": 1,  # 0.1
    "Milímetros (mm)": 0,   # 1
    "Pulgadas (in)": 3,     # 0.001 (aproximadamente 0.254/1000)
    "Pies (ft)": 4          # 0.0001 (más precisión para pies)
}

# Step values para los inputs según la unidad
UNIT_STEP_VALUES = {
    "Metros (m)": 0.01,
    "Centímetros (cm)": 0.1,
    "Milímetros (mm)": 1.0,
    "Pulgadas (in)": 0.001,
    "Pies (ft)": 0.0001
}
UNIT_NAMES = list(UNITS_TO_METERS.keys())
DEFAULT_TUBE_LENGTH_BASE_UNIT = Decimal("6.0")
# Cambiar unidad por defecto a 'Centímetros (cm)'
DEFAULT_UNIT_NAME = "Centímetros (cm)"

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

class OptimizedCalculator: # ... (class definition unchanged) ...
    def __init__(self, precision: int = 28):
        self.precision = precision
        if MPMATH_AVAILABLE: mp.dps = self.precision
    def _calc_radius(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        return sagitta/Decimal('2') + chord**2/(Decimal('8')*sagitta)
    def calculate_radius_standard(self, chord: Decimal, sagitta: Decimal) -> Decimal:
        if sagitta == Decimal('0'): return Decimal('inf')
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
        if not (radius > Decimal('0') and not radius.is_infinite() and not radius.is_nan()): return {"error": "Radio no válido."}
        if not (sagitta > Decimal('0') and sagitta < radius * 2): return {"error": "Sagitta no válida."}
        if not (chord > Decimal('0')): return {"error": "Cuerda no válida."}
        if radius < sagitta : return {"error": "Radio debe ser >= Sagitta."}
        if radius < chord / Decimal('2'): return {"error": "Radio debe ser >= Cuerda/2."}
        h = radius - sagitta
        val_for_acos_dec = h / radius
        if val_for_acos_dec > Decimal('1'): val_for_acos_dec = Decimal('1')
        elif val_for_acos_dec < Decimal('-1'): val_for_acos_dec = Decimal('-1')
        try:
            if MPMATH_AVAILABLE:
                mp.dps = current_prec
                radius_mp = mp.mpf(str(radius))
                val_for_acos_mp = mp.mpf(str(val_for_acos_dec))
                alpha_rad_mp = mp.acos(val_for_acos_mp); theta_rad_mp = 2 * alpha_rad_mp
                arc_length_mp = radius_mp * theta_rad_mp; central_angle_deg_mp = mp.degrees(theta_rad_mp)
                return {"arc_length": Decimal(str(arc_length_mp)), "central_angle_deg": Decimal(str(central_angle_deg_mp)), "error": None}
            else:
                val_for_acos_float = float(val_for_acos_dec)
                alpha_rad_float = math.acos(val_for_acos_float); theta_rad_float = 2 * alpha_rad_float
                arc_length_dec = radius * Decimal(str(theta_rad_float))
                central_angle_deg_dec = Decimal(str(math.degrees(theta_rad_float)))
                return {"arc_length": arc_length_dec, "central_angle_deg": central_angle_dec, "error": None}
        except ValueError as ve: return {"error": f"Error de valor en cálculo de ángulo: {ve}"}
        except Exception as e: return {"error": f"Error inesperado en cálculo de arco: {e}"}

@st.cache_data(ttl=1800, max_entries=10)
def call_gemini_api(prompt: str, api_key: str) -> Dict[str, Any]: # ... (no changes) ...
    if not api_key: return {"success": False, "error": "API key de Gemini no configurada."}
    if not REQUESTS_AVAILABLE: return {"success": False, "error": "Módulo 'requests' no disponible."}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60); response.raise_for_status()
        data = response.json(); candidates = data.get("candidates", [])
        if not candidates: return {"success": False, "error": "Respuesta API: Sin 'candidates'"}
        content = candidates[0].get("content", {}); parts = content.get("parts", [])
        if not parts: return {"success": False, "error": "Respuesta API: Sin 'parts'"}
        text = parts[0].get("text", ""); return {"success": True, "response": text}
    except requests.exceptions.Timeout: return {"success": False, "error": "Error API: Timeout (60s)"}
    except requests.exceptions.HTTPError as e: return {"success": False, "error": f"Error API: {e.response.status_code} ({e.response.reason})"}
    except requests.exceptions.RequestException as e: return {"success": False, "error": f"Error de Conexión API: {str(e)}"}
    except Exception as e: return {"success": False, "error": f"Error procesando API: {str(e)}"}

@st.cache_data(ttl=3600, max_entries=20)
def calculate_radius_all_methods(chord_base_unit_float: float, sagitta_base_unit_float: float, internal_precision: int) -> Dict[str, Any]: # ... (no changes) ...
    calc_context = getcontext(); calc_context.prec = internal_precision
    try:
        chord_base = Decimal(str(chord_base_unit_float)); sagitta_base = Decimal(str(sagitta_base_unit_float))
        arc_length_str, central_angle_str, arc_calc_error_msg = "N/A", "N/A", None
        if chord_base <= Decimal('0') or sagitta_base <= Decimal('0'): return {"error": "Cuerda y Sagitta (en unidad base) deben ser > 0."}
        if sagitta_base >= chord_base / Decimal('2'): return {"error": f"Sagitta ({sagitta_base}) debe ser < mitad de Cuerda ({chord_base/Decimal('2')}) en unidad base."}
        calc_instance = OptimizedCalculator(internal_precision)
        methods_to_run = [("Estándar", calc_instance.calculate_radius_standard),("Segmental", calc_instance.calculate_radius_segmental),("Trigonométrico", calc_instance.calculate_radius_trigonometric)]
        results_decimal = {}
        for name, func in methods_to_run:
            try: results_decimal[name] = func(chord_base, sagitta_base)
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
        if len(valid_results_list) > 1:
            max_dev = max(abs(v - median_radius_base) for v in valid_results_list)
            confidence_dec = Decimal('1') - min(max_dev / median_radius_base, Decimal('1'))
        sag_corr_base = calc_instance.calculate_sagitta_corrected(chord_base, median_radius_base)
        sag_incorr_base = chord_base**2 / (Decimal('2')*median_radius_base) if median_radius_base!=Decimal('0') else Decimal('inf')
        err_perc_dec = Decimal('inf')
        if sag_corr_base.is_finite() and sag_corr_base != Decimal('0'):
            if sag_incorr_base.is_finite(): err_perc_dec = abs(sag_incorr_base - sag_corr_base) / sag_corr_base * Decimal('100')
        elif sag_incorr_base == sag_corr_base: err_perc_dec = Decimal('0')
        return {"success": True, "radius_final_dec_str": str(median_radius_base), "confidence_dec_str": str(confidence_dec), "methods_dec_str": {k: str(v) if isinstance(v,Decimal) else v for k,v in results_decimal.items()}, "sagitta_corrected_dec_str": str(sag_corr_base), "sagitta_incorrect_dec_str": str(sag_incorr_base), "error_percentage_dec_str": str(err_perc_dec), "arc_length_dec_str": arc_length_str, "central_angle_deg_str": central_angle_str, "arc_calculation_error": arc_calc_error_msg }
    except Exception as e: return {"error": f"Error general en núcleo de cálculo: {str(e)}"}

def create_single_arc_visualization(chord_val, sagitta_val, radius_val, plot_title_prefix="Arco", display_precision_cfg=1, unit_name="Unidades"): # ... (no changes) ...
    if not PLOTLY_AVAILABLE: return None
    C, S, R = float(chord_val), float(sagitta_val), float(radius_val)
    if not (R > 1e-9 and S > 1e-9 and C > 1e-9 and R != float('inf') and S < C/2 and R >= S-(1e-9) and R >= C/2-(1e-9)):
        fig = go.Figure(); fig.add_annotation(text=f"Geometría '{plot_title_prefix}' no válida (C={C:.{display_precision_cfg}f}, S={S:.{display_precision_cfg}f}, R={R:.{display_precision_cfg}f} {unit_name}).", xref="paper", yref="paper", showarrow=False); fig.update_layout(height=200, title_text=f"{plot_title_prefix}: No disponible"); return fig
    h_center, k_center = 0.0, S - R; val_for_arccos = np.clip((R - S) / R, -1.0, 1.0); alpha = np.arccos(val_for_arccos)
    t_angles = np.linspace(-alpha, alpha, 100) ; x_arc, y_arc = R * np.sin(t_angles), k_center + R * np.cos(t_angles)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines', name='Arco', line=dict(color='blue', width=2))); fig.add_trace(go.Scatter(x=[-C/2, C/2], y=[0, 0], mode='lines', name='Cuerda', line=dict(color='red', dash='dash'))); fig.add_trace(go.Scatter(x=[0, 0], y=[0, S], mode='lines', name='Sagitta', line=dict(color='green', dash='dash')))
    fig.add_annotation(x=0, y=S * 0.5, text=f"S={S:.{display_precision_cfg}f}", showarrow=False, yshift=10, font=dict(size=10)); fig.add_annotation(x=0, y=-S*0.1, text=f"C={C:.{display_precision_cfg}f}", showarrow=False, yshift=-5, font=dict(size=10))
    fig.update_layout(title_text=f"{plot_title_prefix} (Radio R={R:.{display_precision_cfg}f} {unit_name})", xaxis_title=f"Dimensión X ({unit_name})", yaxis_title=f"Dimensión Y ({unit_name})", yaxis_scaleanchor="x", legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5), margin=dict(t=50, b=120, l=20, r=20), height=500)
    return fig

def main():
    tube_usable_length = None  # Inicialización para evitar errores de variable no definida
    global app_config
    app_config = init_app_config()
    getcontext().prec = app_config['precision']
    display_prec_cfg = app_config['display_precision_general']

    st.title("🔢 Calculadora de Precisión para Arcos y Tubos")

    # --- Session State Initialization ---
    if 'selected_unit_name' not in st.session_state:
        st.session_state.selected_unit_name = DEFAULT_UNIT_NAME
    if 'cantidad_arcos_widget' not in st.session_state: # MODIFIED KEY
        st.session_state.cantidad_arcos_widget = 1

    if 'chord_input_float' not in st.session_state:
        st.session_state.chord_input_float = 1000.0
    if 'sagitta_input_float' not in st.session_state:
        st.session_state.sagitta_input_float = 250.0
    if 'tube_length_input_float' not in st.session_state:
        if st.session_state.selected_unit_name == "Centímetros (cm)":
            st.session_state.tube_length_input_float = 600.0
        else:
            current_unit_factor = UNITS_TO_METERS[st.session_state.selected_unit_name]
            st.session_state.tube_length_input_float = float(600.0 * float(UNITS_TO_METERS["Centímetros (cm)"]) / float(current_unit_factor))

    # --- Entradas adicionales ---
    if 'perfil_tub_input' not in st.session_state:
        # Inicializar según la unidad por defecto (Centímetros)
        if st.session_state.selected_unit_name == "Centímetros (cm)":
            st.session_state.perfil_tub_input = 4.8 * float(UNITS_TO_METERS["Centímetros (cm)"])
        else:
            st.session_state.perfil_tub_input = 0.048
    if 'tam_regla_input' not in st.session_state:
        # Inicializar según la unidad por defecto (Centímetros)
        if st.session_state.selected_unit_name == "Centímetros (cm)":
            st.session_state.tam_regla_input = 216.0 * float(UNITS_TO_METERS["Centímetros (cm)"])
        else:
            st.session_state.tam_regla_input = 2.16
    if 'anc_regla_input' not in st.session_state:
        # Inicializar según la unidad por defecto (Centímetros)
        if st.session_state.selected_unit_name == "Centímetros (cm)":
            st.session_state.anc_regla_input = 3.781 * float(UNITS_TO_METERS["Centímetros (cm)"])
        else:
            st.session_state.anc_regla_input = 0.03781
    if 'num_arcos_input' not in st.session_state:
        st.session_state.num_arcos_input = 1
    if 'despedico_input' not in st.session_state:
        # Despedico siempre en centímetros
        st.session_state.despedico_input = 40.0 * float(UNITS_TO_METERS["Centímetros (cm)"])

    newly_selected_unit_name = st.selectbox("Unidad para Entradas/Resultados:", UNIT_NAMES, index=UNIT_NAMES.index(st.session_state.selected_unit_name), key="unit_selector_widget")

    if newly_selected_unit_name != st.session_state.selected_unit_name:
        old_unit_factor = UNITS_TO_METERS[st.session_state.selected_unit_name]
        new_unit_factor = UNITS_TO_METERS[newly_selected_unit_name]

        # Convertir todos los valores relevantes a la nueva unidad
        st.session_state.chord_input_float = st.session_state.chord_input_float * float(old_unit_factor) / float(new_unit_factor)
        st.session_state.sagitta_input_float = st.session_state.sagitta_input_float * float(old_unit_factor) / float(new_unit_factor)
        st.session_state.tube_length_input_float = st.session_state.tube_length_input_float * float(old_unit_factor) / float(new_unit_factor)
        st.session_state.perfil_tub_input = st.session_state.perfil_tub_input * float(old_unit_factor) / float(new_unit_factor)
        st.session_state.tam_regla_input = st.session_state.tam_regla_input * float(old_unit_factor) / float(new_unit_factor)
        st.session_state.anc_regla_input = st.session_state.anc_regla_input * float(old_unit_factor) / float(new_unit_factor)
        # Despedico NO se convierte, siempre permanece en centímetros

        st.session_state.selected_unit_name = newly_selected_unit_name
        st.rerun()

    selected_unit_name_for_display = st.session_state.selected_unit_name
    factor_to_base_unit = UNITS_TO_METERS[selected_unit_name_for_display]

    st.info(f"Precisión interna: {app_config['precision']} dígitos. Entradas/Resultados en {selected_unit_name_for_display}. Resultados mostrados con {display_prec_cfg} decimal.")

    with st.sidebar:
        st.header("⚙️ Información de la App"); st.write(f"✅ mpmath: {'Disponible' if MPMATH_AVAILABLE else 'No disponible'}"); st.write(f"✅ Plotly: {'Disponible' if PLOTLY_AVAILABLE else 'No disponible'}")
        if REQUESTS_AVAILABLE: st.write(f"✅ API Gemini: {'Configurada' if app_config['gemini_api_key'] else 'Key no configurada'}")
        else: st.write("❌ API Gemini: No disponible ('requests' faltante)")
        st.write(f"⚙️ Precisión Decimal: {getcontext().prec} dígitos"); st.markdown("---"); st.caption("Calculadora geométrica.")

    st.header("📐 Entradas para Cálculo")

    # Use 'cantidad_arcos_widget' as key and for value persistence
    st.number_input(
        "Cantidad de Arcos a cubrir",
        min_value=1,
        step=1,
        format="%d",
        key="cantidad_arcos_widget",
        help="Número total de arcos idénticos a cubrir con los tubos."
    )

    if 'example_values_float' in st.session_state:
        ex_data = st.session_state.example_values_float
        st.session_state.selected_unit_name = UNIT_NAMES[0]
        st.session_state.cantidad_arcos_widget = 1 # MODIFIED KEY
        st.session_state.chord_input_float = ex_data.get('chord', 1000.0)
        st.session_state.sagitta_input_float = ex_data.get('sagitta', 250.0)
        meter_unit_factor = UNITS_TO_METERS[UNIT_NAMES[0]]
        st.session_state.tube_length_input_float = ex_data.get('tube_length', 600.0)
        del st.session_state.example_values_float; st.rerun()

    # Usar precisión específica según la unidad seleccionada
    unit_precision = UNIT_DECIMAL_PRECISION[selected_unit_name_for_display]
    unit_step = UNIT_STEP_VALUES[selected_unit_name_for_display]
    num_input_fmt_str = f"%.{unit_precision}f"

    st.session_state.chord_input_float = st.number_input(f"Cuerda (c) en {selected_unit_name_for_display}", min_value=1e-9, max_value=1e12, value=st.session_state.chord_input_float, step=unit_step, format=num_input_fmt_str, key="chord_input_widget", help=f"Longitud de la cuerda del arco.")
    st.session_state.sagitta_input_float = st.number_input(f"Sagitta/Flecha (s) en {selected_unit_name_for_display}", min_value=1e-9, max_value=1e12, value=st.session_state.sagitta_input_float, step=unit_step, format=num_input_fmt_str, key="sagitta_input_widget", help=f"Altura máxima del arco.")
    st.session_state.tube_length_input_float = st.number_input(f"Longitud Tubo (L_tubo) en {selected_unit_name_for_display}", min_value=0.0, max_value=1e12, value=st.session_state.tube_length_input_float, step=unit_step, format=num_input_fmt_str, key="tube_length_input_widget", help=f"Longitud del tubo a rolar (opcional).")

    # --- Entradas adicionales según app de referencia ---
    st.subheader("Entradas adicionales (opcional)")
    # Perfil de tubo (decimal, ej: 4.8 cm)
    perfil_tub_val = st.session_state.perfil_tub_input / float(factor_to_base_unit)
    perfil_tub_val = st.number_input(
        f"Perfil Tubo (perfil tub.) en {selected_unit_name_for_display}",
        min_value=0.0, max_value=100.0, value=perfil_tub_val, step=unit_step, format=num_input_fmt_str,
        key="perfil_tub_input_widget",
        help=f"Perfil del tubo en {selected_unit_name_for_display}")
    st.session_state.perfil_tub_input = perfil_tub_val * float(factor_to_base_unit)

    # Tamaño de regla
    tam_regla_val = st.session_state.tam_regla_input / float(factor_to_base_unit)
    tam_regla_val = st.number_input(
        f"Tamaño de Regla (tam. regla) en {selected_unit_name_for_display}",
        min_value=0.0, max_value=1e8, value=tam_regla_val, step=1.0,
        key="tam_regla_input_widget",
        help=f"Tamaño de la regla utilizada en {selected_unit_name_for_display}")
    st.session_state.tam_regla_input = tam_regla_val * float(factor_to_base_unit)

    # Ancho de regla
    anc_regla_val = st.session_state.anc_regla_input / float(factor_to_base_unit)
    # Ajustar el valor si está fuera del rango permitido
    if anc_regla_val > 10000.0:
        anc_regla_val = 10000.0
        st.session_state.anc_regla_input = anc_regla_val * float(factor_to_base_unit)
    elif anc_regla_val < 0.0:
        anc_regla_val = 0.0
        st.session_state.anc_regla_input = anc_regla_val * float(factor_to_base_unit)
    
    anc_regla_val = st.number_input(
        f"Ancho de Regla (anc. regla) en {selected_unit_name_for_display}",
        min_value=0.0, max_value=10000.0, value=anc_regla_val, step=0.001,
        key="anc_regla_input_widget",
        help=f"Ancho de la regla utilizada en {selected_unit_name_for_display}")
    st.session_state.anc_regla_input = anc_regla_val * float(factor_to_base_unit)

    # Número de arcos
    st.session_state.num_arcos_input = st.number_input(
        "Número de Arcos (num. arcos)",
        min_value=1, max_value=1000, value=st.session_state.num_arcos_input, step=1,
        key="num_arcos_input_widget",
        help="Número de arcos a calcular")

    # Despedico (siempre en centímetros, no se convierte)
    despedico_val_cm = st.session_state.despedico_input / float(UNITS_TO_METERS["Centímetros (cm)"])
    if 'despedico_input' not in st.session_state:
        despedico_val_cm = 40.0
        st.session_state.despedico_input = despedico_val_cm * float(UNITS_TO_METERS["Centímetros (cm)"])
    
    despedico_val_cm = st.number_input(
        "Despedico (cm)",
        min_value=0.0, max_value=10000.0, value=despedico_val_cm, step=0.01,
        key="despedico_input_widget",
        help="Despedico en centímetros aplicado al cálculo.")
    st.session_state.despedico_input = despedico_val_cm * float(UNITS_TO_METERS["Centímetros (cm)"])
    # --- Fin de entradas adicionales ---
    # Aquí se pueden agregar los cálculos específicos usando estos campos cuando se disponga de las fórmulas.
    # Por ejemplo: calcular flecha del tubo, cuerda del tubo, flecha de la regla, desarrollo de arco, etc.
    # TODO: Implementar lógica de cálculo adicional según fórmulas específicas de la app original.

    # Call the calculation and display function on every rerun
    perform_calculations_and_display()

def perform_calculations_and_display():
    # Acceso a variables desde el estado de sesión y definición local de factor_to_base_unit
    selected_unit_name_for_display = st.session_state.selected_unit_name
    factor_to_base_unit = UNITS_TO_METERS[selected_unit_name_for_display]
    # Usar precisión específica según la unidad seleccionada para mostrar resultados
    display_prec_cfg = UNIT_DECIMAL_PRECISION[selected_unit_name_for_display]

    current_chord_f_selected_unit = st.session_state.chord_input_float
    current_sagitta_f_selected_unit = st.session_state.sagitta_input_float
    current_tube_len_f_selected_unit = st.session_state.tube_length_input_float
    cantidad_arcos_val = st.session_state.cantidad_arcos_widget # MODIFIED KEY

    chord_base_f = current_chord_f_selected_unit * float(factor_to_base_unit)
    sagitta_base_f = current_sagitta_f_selected_unit * float(factor_to_base_unit)

    if current_chord_f_selected_unit <= 1e-9 or current_sagitta_f_selected_unit <= 1e-9: st.error("❌ Cuerda y Sagitta deben ser > 0.")
    elif Decimal(str(current_sagitta_f_selected_unit)) >= Decimal(str(current_chord_f_selected_unit)) / Decimal('2'):
        st.error(f"❌ Sagitta (s={current_sagitta_f_selected_unit:.{display_prec_cfg}f}) debe ser < mitad de Cuerda (c/2 = {current_chord_f_selected_unit/2:.{display_prec_cfg}f}).")
    elif current_tube_len_f_selected_unit < 0.0 and abs(current_tube_len_f_selected_unit) > 1e-9 : st.error("❌ Longitud del Tubo no puede ser negativa (0 si no se usa).")
    elif cantidad_arcos_val < 1: st.error("❌ Cantidad de Arcos debe ser al menos 1.")
    else:
        with st.spinner("🧠 Calculando..."):
            start_time = time.time()
            calc_results = calculate_radius_all_methods(chord_base_f, sagitta_base_f, app_config['precision'])
            computation_time = time.time() - start_time

            if calc_results.get("success"):
                radius_base_dec = Decimal(calc_results['radius_final_dec_str'])
                arc_length_base_str = calc_results.get('arc_length_dec_str', "N/A")
                central_angle_base_str = calc_results.get('central_angle_deg_str', "N/A")
                arc_calc_err = calc_results.get('arc_calculation_error')

                radius_display = radius_base_dec / factor_to_base_unit
                arc_length_display_one_arc, central_angle_display = None, None
                if arc_length_base_str != "N/A" and not arc_calc_err:
                    try: arc_length_display_one_arc = Decimal(arc_length_base_str) / factor_to_base_unit
                    except: arc_calc_err = arc_calc_err or "Error al convertir long. arco para display"
                if central_angle_base_str != "N/A" and not arc_calc_err:
                    try: central_angle_display = Decimal(central_angle_base_str)
                    except: arc_calc_err = arc_calc_err or "Error al convertir ángulo para display"

                # Guardar todos los resultados calculados en session_state para persistencia
                st.session_state.radius_display = radius_display
                st.session_state.arc_length_display_one_arc = arc_length_display_one_arc
                st.session_state.central_angle_display = central_angle_display
                st.session_state.arc_calc_err = arc_calc_err
                st.session_state.confidence_dec = Decimal(calc_results['confidence_dec_str'])
                st.session_state.methods_ui_data = [{"Método": k, f"Resultado (R)": f"{(Decimal(v)/factor_to_base_unit):.{display_prec_cfg}f}" if not v.startswith("Error") else v, "Estado": "✅" if not v.startswith("Error") else "❌"} for k,v in calc_results['methods_dec_str'].items()]
                st.session_state.computation_time = computation_time
                st.session_state.sag_corr_base_dec = Decimal(calc_results['sagitta_corrected_dec_str'])
                st.session_state.sag_incorr_base_dec = Decimal(calc_results['sagitta_incorrect_dec_str'])
                st.session_state.err_perc_dec = Decimal(calc_results['error_percentage_dec_str'])

                # 1. MOSTRAR RESULTADO PRINCIPAL DEL RADIO
                st.success(f"✅ **Radio del Arco Principal (R): {radius_display:.{display_prec_cfg}f} {selected_unit_name_for_display}**")

                # 2. CÁLCULO DE FLECHA PARA TUBO (PRIMERA PRIORIDAD)
                flecha_tubo_calc_display = None
                if current_tube_len_f_selected_unit > 1e-9:
                    st.subheader("🏹 Cálculo de Flecha para Tubo")
                    ui_calc = OptimizedCalculator(app_config['precision'])
                    if radius_base_dec.is_finite() and radius_base_dec > Decimal('0'):
                        tube_len_base_dec = Decimal(str(current_tube_len_f_selected_unit)) * factor_to_base_unit
                        flecha_tubo_base_dec = ui_calc.calculate_sagitta_corrected(tube_len_base_dec, radius_base_dec)
                        flecha_tubo_calc_display = flecha_tubo_base_dec / factor_to_base_unit
                        col_t1,col_t2=st.columns(2); col_t1.metric(f"Longitud Tubo ({selected_unit_name_for_display})",f"{current_tube_len_f_selected_unit:.{app_config['display_precision_metrics']}f}"); col_t2.metric(f"Flecha Tubo Calculada ({selected_unit_name_for_display})",f"{flecha_tubo_calc_display:.{display_prec_cfg}f}")
                        st.caption(f"Valores en {selected_unit_name_for_display}")
                    else: 
                        st.warning("No se calcula flecha de tubo: radio del arco principal no válido.")

                # 3. AJUSTE DE TUBOS EN EL ARCO (SEGUNDA PRIORIDAD)
                num_tubes_output_str = "N/A"
                if current_tube_len_f_selected_unit > 1e-9 and arc_length_display_one_arc is not None and arc_length_display_one_arc > Decimal('1e-7') and not arc_calc_err:
                    total_arc_length_for_tubes_display = arc_length_display_one_arc * cantidad_arcos_val
                    tube_len_selected_unit_dec = Decimal(str(current_tube_len_f_selected_unit))
                    if tube_len_selected_unit_dec > Decimal('1e-9'):
                        num_tubes_precise = total_arc_length_for_tubes_display / tube_len_selected_unit_dec
                        full_tubes = math.floor(num_tubes_precise)
                        remainder_fraction = num_tubes_precise - Decimal(str(full_tubes))
                        remainder_length_display = remainder_fraction * tube_len_selected_unit_dec
                        num_tubes_output_str = f"{full_tubes} entero(s) y un segmento de {remainder_length_display:.{display_prec_cfg}f} {selected_unit_name_for_display}"
                    else:
                        num_tubes_output_str = "Longitud de tubo inválida para cálculo de ajuste."
                        full_tubes = 0; remainder_length_display = Decimal('0')

                    st.subheader("🧩 Ajuste de Tubos en el Arco")
                    st.write(f"Cantidad de Arcos Considerada: **{cantidad_arcos_val}**")
                    st.write(f"Longitud Total de Arco a Cubrir: **{total_arc_length_for_tubes_display:.{display_prec_cfg}f} {selected_unit_name_for_display}**")
                    st.write(f"Longitud de cada Tubo: **{current_tube_len_f_selected_unit:.{app_config['display_precision_metrics']}f} {selected_unit_name_for_display}**")
                    if "entero(s)" in num_tubes_output_str :
                         st.metric(label="Resultado del Ajuste", value=f"{full_tubes} tubos + {remainder_length_display:.{display_prec_cfg}f} {selected_unit_name_for_display}")
                         st.caption(f"El segmento adicional mide {remainder_length_display:.{display_prec_cfg}f} {selected_unit_name_for_display} (equivale a {remainder_fraction:.{display_prec_cfg}f} del largo de un tubo).")
                    else: st.warning(num_tubes_output_str)
                elif arc_calc_err and current_tube_len_f_selected_unit > 1e-9: 
                    st.caption(f"Ajuste de tubos no calculado (error en L_arco): {arc_calc_err}")

                # 4. DIMENSIONES DEL ARCO PRINCIPAL
                if arc_calc_err: st.warning(f"Info cálculo de arco: {arc_calc_err}")
                elif arc_length_display_one_arc is not None and arc_length_display_one_arc > Decimal('1e-7'):
                    st.subheader("📏 Dimensiones del Arco Principal (por unidad)")
                    col_L, col_A = st.columns(2)
                    col_L.metric(f"Longitud de 1 Arco (L_arco)", f"{arc_length_display_one_arc:.{display_prec_cfg}f} {selected_unit_name_for_display}")
                    if central_angle_display is not None: col_A.metric("Ángulo Central (por arco)", f"{central_angle_display:.{display_prec_cfg}f}°")

                # 5. MÉTRICAS DE CONFIANZA Y TIEMPO
                confidence_dec = Decimal(calc_results['confidence_dec_str'])
                col_m1,col_m2,col_m3=st.columns(3); col_m1.metric("Confianza (Radio)",f"{confidence_dec:.2%}")
                valid_m_cnt=sum(1 for v in calc_results['methods_dec_str'].values() if not v.startswith("Error"))
                col_m2.metric("Métodos Válidos",f"{valid_m_cnt}/{len(calc_results['methods_dec_str'])}"); col_m3.metric("Tiempo Cálculo",f"{computation_time*1000:.1f}ms")

                # 6. DETALLES DEL CÁLCULO DEL RADIO
                st.subheader(f"Detalles del Cálculo del Radio (en {selected_unit_name_for_display})")
                methods_ui_data = [{"Método": k, f"Resultado (R)": f"{(Decimal(v)/factor_to_base_unit):.{display_prec_cfg}f}" if not v.startswith("Error") else v, "Estado": "✅" if not v.startswith("Error") else "❌"} for k,v in calc_results['methods_dec_str'].items()]
                st.table(methods_ui_data)

                # 7. VERIFICACIÓN DE SAGITTA
                sag_corr_base_dec = Decimal(calc_results['sagitta_corrected_dec_str'])
                sag_incorr_base_dec = Decimal(calc_results['sagitta_incorrect_dec_str'])
                err_perc_dec = Decimal(calc_results['error_percentage_dec_str'])
                sag_corr_display = sag_corr_base_dec / factor_to_base_unit
                sag_incorr_display = sag_incorr_base_dec / factor_to_base_unit if sag_incorr_base_dec.is_finite() else Decimal('inf')
                st.subheader(f"Verificación de Sagitta (Arco Principal)")
                col_s1,col_s2=st.columns(2); col_s1.metric(f"s con L²/(8R)",f"{sag_corr_display:.{display_prec_cfg}f} {selected_unit_name_for_display}"); col_s2.metric(f"s con L²/(2R)",f"{sag_incorr_display:.{display_prec_cfg}f} {selected_unit_name_for_display}")
                if err_perc_dec.is_finite(): st.info(f"Error relativo (sagitta): {err_perc_dec:.1f}%")
                else: st.info("Error relativo (sagitta): Indefinido.")

                # 8. VISUALIZACIONES (AL FINAL)
                st.subheader("📊 Visualización del Arco Principal")
                main_arc_plot_fig = create_single_arc_visualization(Decimal(str(current_chord_f_selected_unit)), Decimal(str(current_sagitta_f_selected_unit)), radius_display, plot_title_prefix="Arco Principal", display_precision_cfg=display_prec_cfg, unit_name=selected_unit_name_for_display)
                if main_arc_plot_fig and PLOTLY_AVAILABLE: st.plotly_chart(main_arc_plot_fig, use_container_width=True)

                # 9. VISUALIZACIÓN DEL TUBO ROLADO (SI APLICA)
                if flecha_tubo_calc_display is not None and current_tube_len_f_selected_unit > 1e-9:
                    st.subheader("📊 Visualización del Tubo Rolado")
                    tube_arc_plot_fig = create_single_arc_visualization(Decimal(str(current_tube_len_f_selected_unit)), flecha_tubo_calc_display, radius_display, plot_title_prefix="Tubo Rolado", display_precision_cfg=display_prec_cfg, unit_name=selected_unit_name_for_display)
                    if tube_arc_plot_fig and PLOTLY_AVAILABLE: st.plotly_chart(tube_arc_plot_fig, use_container_width=True)

                # 10. ANÁLISIS CON IA (AL FINAL)
                if app_config['gemini_api_key'] and REQUESTS_AVAILABLE:
                    with st.expander("🤖 Análisis con IA (Opcional)", expanded=False):
                        with st.spinner("🧠 Consultando IA..."):
                            s_tubo_str_ai = f"{flecha_tubo_calc_display:.{display_prec_cfg}f} {selected_unit_name_for_display}" if flecha_tubo_calc_display is not None else "N/A"
                            arc_len_str_ai = f"{arc_length_display_one_arc:.{display_prec_cfg}f} {selected_unit_name_for_display}" if arc_length_display_one_arc is not None and arc_length_display_one_arc > Decimal('1e-7') and not arc_calc_err else "N/A"
                            central_angle_str_ai = f"{central_angle_display:.{display_prec_cfg}f}°" if central_angle_display is not None and not arc_calc_err else "N/A"
                            total_arc_len_for_ai = f"{(arc_length_display_one_arc * cantidad_arcos_val):.{display_prec_cfg}f} {selected_unit_name_for_display}" if arc_length_display_one_arc is not None and arc_length_display_one_arc > Decimal('1e-7') and cantidad_arcos_val >=1 and not arc_calc_err else "N/A"

                            prompt_lines = []
                            prompt_lines.append(f"Análisis de cálculo de arco (unidades de entrada/salida: {selected_unit_name_for_display}):")
                            prompt_lines.append(f"- Cuerda (c): {current_chord_f_selected_unit:.{app_config['display_precision_metrics']}f}, Sagitta (s): {current_sagitta_f_selected_unit:.{app_config['display_precision_metrics']}f}, Radio calculado (R): {radius_display:.{display_prec_cfg}f}.")
                            prompt_lines.append(f"- Para 1 arco: Longitud (L_arco): {arc_len_str_ai}, Ángulo Central: {central_angle_str_ai}.")
                            prompt_lines.append(f"- Cantidad de Arcos: {cantidad_arcos_val}, L_arco Total: {total_arc_len_for_ai}.")
                            prompt_lines.append(f"¿Es R geométricamente coherente con c y s? ¿Son L_arco y Ángulo coherentes?")
                            prompt_lines.append(f"Si L_tubo = {current_tube_len_f_selected_unit:.{app_config['display_precision_metrics']}f} (>0), y s_tubo = {s_tubo_str_ai}, ¿es s_tubo coherente?")
                            prompt_lines.append(f"Para L_arco Total y L_tubo, ¿cuántos tubos se necesitan ({num_tubes_output_str})?")
                            prompt_lines.append(f"Explicación concisa.")
                            ai_prompt = "\n".join(prompt_lines)

                            ai_response = call_gemini_api(ai_prompt, app_config['gemini_api_key'])
                            if ai_response["success"]: st.info("Respuesta de IA:"); st.markdown(ai_response["response"])
                            else: st.error(f"Error IA: {ai_response['error']}")
                elif not app_config['gemini_api_key'] and REQUESTS_AVAILABLE:
                     with st.expander("🤖 Análisis con IA (Opcional)", expanded=False): st.warning("API Key de Gemini no configurada.")

                # Guardar todos los resultados calculados en session_state para persistencia
                st.session_state.radius_display = radius_display
                st.session_state.arc_length_display_one_arc = arc_length_display_one_arc
                st.session_state.central_angle_display = central_angle_display
                st.session_state.arc_calc_err = arc_calc_err
                st.session_state.confidence_dec = confidence_dec
                st.session_state.methods_ui_data = methods_ui_data
                st.session_state.computation_time = computation_time
                st.session_state.sag_corr_display = sag_corr_display
                st.session_state.sag_incorr_display = sag_incorr_display
                st.session_state.err_perc_dec = err_perc_dec
                st.session_state.flecha_tubo_calc_display = flecha_tubo_calc_display
                st.session_state.num_tubes_output_str = num_tubes_output_str
                
                # Guardar datos de tubos si fueron calculados
                try:
                    if 'total_arc_length_for_tubes_display' in locals():
                        st.session_state.total_arc_length_for_tubes_display = total_arc_length_for_tubes_display
                    if 'full_tubes' in locals():
                        st.session_state.full_tubes = full_tubes
                    if 'remainder_length_display' in locals():
                        st.session_state.remainder_length_display = remainder_length_display
                except NameError:
                    pass

            else:
                st.error(f"❌ {calc_results.get('error', 'Error en cálculo principal.')}")

    # --- Bloque de Chat Asistente IA (AL FINAL) ---
    st.markdown("---")
    st.header("💬 Asistente IA (Gemini Flash 2)")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = {}
    if st.button("🔄 Resetear chat y contexto", key="reset_chat_btn", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chat_context = {}
        st.rerun()

    # Sugerencias dinámicas según contexto
    sugerencias = [
        "¿Cuántos tubos necesito para cubrir todos los arcos?",
        "¿Cómo se calcula el radio con los datos actuales?",
        "Muéstrame el gráfico del arco principal.",
        "¿Puedo cambiar la longitud del tubo a 100?",
        "Explica de forma sencilla cómo se obtiene la sagitta."
    ]
    # Si hay cálculos, sugerencias más específicas
    if 'radius_display' in st.session_state:
        sugerencias.insert(0, f"¿Por qué el radio calculado es {st.session_state.radius_display:.1f}?")
    if 'num_tubes_output_str' in st.session_state:
        sugerencias.insert(0, f"¿Qué significa: {st.session_state.num_tubes_output_str}?")

    st.caption("Sugerencias rápidas:")
    cols_sug = st.columns(len(sugerencias))
    for idx, sug in enumerate(sugerencias):
        if cols_sug[idx].button(sug, key=f"sug_{idx}", use_container_width=True):
            st.session_state.chat_input = sug
            st.rerun()

    # Entrada de usuario
    chat_input = st.text_input("Escribe tu pregunta o instrucción:", value=st.session_state.get('chat_input', ''), key="chat_input_box")
    enviar = st.button("Enviar", key="enviar_chat_btn", use_container_width=True)

    # Procesar mensaje
    if enviar and chat_input.strip():
        # Construir contexto para IA con TODOS los datos de entrada actuales y resultados
        contexto = []
        contexto.append(f"Unidad seleccionada: {st.session_state.get('selected_unit_name', 'N/A')}")
        contexto.append(f"Cuerda (c): {st.session_state.get('chord_input_float', 'N/A')}")
        contexto.append(f"Sagitta (s): {st.session_state.get('sagitta_input_float', 'N/A')}")
        contexto.append(f"Longitud de tubo (L_tubo): {st.session_state.get('tube_length_input_float', 'N/A')}")
        contexto.append(f"Cantidad de arcos: {st.session_state.get('cantidad_arcos_widget', 'N/A')}")
        if 'radius_display' in st.session_state:
            contexto.append(f"Radio calculado: {st.session_state.radius_display}")
        if 'arc_length_display_one_arc' in st.session_state:
            contexto.append(f"Longitud de arco (1 arco): {st.session_state.arc_length_display_one_arc}")
        if 'num_tubes_output_str' in st.session_state:
            contexto.append(f"Resultado de ajuste de tubos: {st.session_state.num_tubes_output_str}")
        # Incluir historial si se desea contexto
        historial = "\n".join([f"Usuario: {h['user']}\nAsistente: {h['ai']}" for h in st.session_state.chat_history])
        prompt = (
            f"Historial previo:\n{historial}\n" if historial else ""
        ) + (
            "\n".join(contexto) + "\n" if contexto else ""
        ) + f"\nPregunta o instrucción del usuario: {chat_input}\n"

        # Llamar a Gemini Flash 2
        ai_response = call_gemini_api(prompt, app_config['gemini_api_key'])
        if ai_response["success"]:
            respuesta = ai_response["response"]
            # Detectar si el usuario quiere cambiar algún valor
            # (Ejemplo simple: "cambiar cuerda a 50")
            import re
            cambios = []
            for campo, key, tipo in [
                ("cuerda", "chord_input_float", float),
                ("sagitta", "sagitta_input_float", float),
                ("longitud de tubo", "tube_length_input_float", float),
                ("cantidad de arcos", "cantidad_arcos_widget", int)
            ]:
                patron = rf"cambiar {campo} a ([0-9]+(\.[0-9]+)?)"
                m = re.search(patron, chat_input, re.IGNORECASE)
                if m:
                    nuevo_valor = tipo(m.group(1))
                    st.session_state[key] = nuevo_valor
                    cambios.append(f"{campo} cambiado a {nuevo_valor}")
            st.session_state.chat_history.append({"user": chat_input, "ai": respuesta})
            st.markdown(f"<div style='background:#23272f;color:#f5f5fa;padding:0.5em 1em;border-radius:8px;margin-bottom:0.5em;'><b>Asistente:</b> {respuesta}</div>", unsafe_allow_html=True)
            if cambios:
                st.success("; ".join(cambios))
        else:
            st.session_state.chat_history.append({"user": chat_input, "ai": f"❌ {ai_response['error']}"})
            st.session_state.chat_input = chat_input
        st.rerun()

    # Mostrar historial de chat
    for h in st.session_state.chat_history[-8:]:
        st.markdown(f"<div style='background:#23272f;color:#f5f5fa;padding:0.5em 1em;border-radius:8px;margin-bottom:0.5em;'><b>Usuario:</b> {h['user']}<br><b>Asistente:</b> {h['ai']}</div>", unsafe_allow_html=True)

    st.subheader("📋 Ejemplos")
    examples_float_data = [{"name": "Arco Estándar", "chord": 100.0, "sagitta": 10.0, "tube_length": 50.0},{"name": "Puente Pequeño", "chord": 50.0, "sagitta": 5.0},{"name": "Lente Óptica", "chord": 10.0, "sagitta": 0.5, "tube_length": 8.0},{"name": "Curva Suave", "chord": 1000.0, "sagitta": 25.0, "tube_length": 200.0}]
    cols_ex = st.columns(len(examples_float_data));
    for idx, ex_data in enumerate(examples_float_data):
        ex_btn_label = f"📐 {ex_data['name']}"; ex_btn_help = f"C: {ex_data['chord']}, S: {ex_data['sagitta']}"
        if 'tube_length' in ex_data: ex_btn_help += f", L_tubo: {ex_data['tube_length']}"
        if cols_ex[idx].button(ex_btn_label, key=f"ex_btn_{idx}", help=ex_btn_help, use_container_width=True):
            st.session_state.example_values_float = ex_data
            st.session_state.selected_unit_name = UNIT_NAMES[0]
            st.session_state.cantidad_arcos_widget = 1 # Corrected key
            st.session_state.chord_input_float = ex_data.get('chord', 1000.0)
            st.session_state.sagitta_input_float = ex_data.get('sagitta', 250.0)
            default_tube_val = float(DEFAULT_TUBE_LENGTH_BASE_UNIT / UNITS_TO_METERS[st.session_state.selected_unit_name])
            st.session_state.tube_length_input_float = ex_data.get('tube_length', default_tube_val)
            st.rerun()

    st.markdown("---"); st.markdown("<div style='text-align:center;color:#666;margin-top:1rem'><small>Calculadora de Precisión Arcos/Tubos</small></div>", unsafe_allow_html=True)

    js_script = (
        '<script>\n'
        'try {\n'
        '    const numberInputs = document.querySelectorAll(\'input[type="number"]\');\n'
        '    numberInputs.forEach(function(input) {\n'
        '        input.addEventListener(\'focus\', function() { this.select(); });\n'
        '    });\n'
        '} catch (e) { /* console.error(\'Error attaching focus listeners:\', e); */ }\n'
        '</script>'
    )
    st.markdown(js_script, unsafe_allow_html=True)

    # --- Cálculos estándar de geometría de arcos y tubos ---
    # Variables de entrada
    # tam_tub = st.session_state.tam_tub_input  # ELIMINADO
    perfil_tub = st.session_state.perfil_tub_input
    tam_regla = st.session_state.tam_regla_input
    anc_regla = st.session_state.anc_regla_input
    num_arcos = st.session_state.num_arcos_input
    despedico = st.session_state.despedico_input
    # Usar solo tube_length_input_float para cálculos de tubo
    tam_tub = st.session_state.tube_length_input_float * float(factor_to_base_unit)

    # Cálculos básicos
    # 1. Flecha del tubo (usando cuerda y radio)
    try:
        cuerda_tubo = tam_tub  # Ahora el tamaño de tubo es la longitud de tubo
        radio = st.session_state.radius_display if 'radius_display' in st.session_state else None
        if radio is not None and cuerda_tubo > 0 and radio > cuerda_tubo/2:
            flecha_tubo = radio - ((radio**2 - (cuerda_tubo/2)**2)**0.5)
        else:
            flecha_tubo = None
    except Exception:
        flecha_tubo = None

    # 2. Cuerda del tubo (usando flecha y radio)
    try:
        if radio is not None and flecha_tubo is not None and flecha_tubo > 0:
            cuerda_tubo_calc = 2 * ((2*radio*flecha_tubo - flecha_tubo**2)**0.5)
        else:
            cuerda_tubo_calc = None
    except Exception:
        cuerda_tubo_calc = None

    # 3. Flecha de la regla (usando tam_regla y radio)
    try:
        if radio is not None and tam_regla > 0 and radio > tam_regla/2:
            flecha_regla = radio - ((radio**2 - (tam_regla/2)**2)**0.5)
        else:
            flecha_regla = None
    except Exception:
        flecha_regla = None

    # 4. Desarrollo de arco (longitud de arco para cuerda y radio)
    try:
        if radio is not None and cuerda_tubo > 0 and radio > cuerda_tubo/2:
            angulo_rad = 2 * math.asin(cuerda_tubo/(2*radio))
            desarrollo_arco = radio * angulo_rad
        else:
            desarrollo_arco = None
    except Exception:
        desarrollo_arco = None

    # 5. Diámetro al eje
    try:
        if radio is not None:
            diametro_eje = 2 * radio
        else:
            diametro_eje = None
    except Exception:
        diametro_eje = None

    # 6. Cantidad de tubos (longitud total de arco / tamaño de tubo)
    try:
        if desarrollo_arco is not None and tam_tub > 0:
            cantidad_tubos = desarrollo_arco / tam_tub
        else:
            cantidad_tubos = None
    except Exception:
        cantidad_tubos = None

    # --- Mostrar resultados en la interfaz ---
    st.markdown("## Resultados geométricos estándar")
    col1, col2 = st.columns(2)
    col1.markdown(f"**Flecha del tubo:** {flecha_tubo:.3f}" if flecha_tubo is not None else "**Flecha del tubo:** N/A")
    col2.markdown(f"**Cuerda del tubo:** {cuerda_tubo_calc:.3f}" if cuerda_tubo_calc is not None else "**Cuerda del tubo:** N/A")
    col1.markdown(f"**Flecha de la regla:** {flecha_regla:.3f}" if flecha_regla is not None else "**Flecha de la regla:** N/A")
    col2.markdown(f"**Desarrollo de arco:** {desarrollo_arco:.3f}" if desarrollo_arco is not None else "**Desarrollo de arco:** N/A")
    col1.markdown(f"**Diámetro al eje:** {diametro_eje:.3f}" if diametro_eje is not None else "**Diámetro al eje:** N/A")
    col2.markdown(f"**Cantidad de tubos:** {cantidad_tubos:.3f}" if cantidad_tubos is not None else "**Cantidad de tubos:** N/A")

if __name__ == "__main__":
    if 'selected_unit_name' not in st.session_state:
        st.session_state.selected_unit_name = DEFAULT_UNIT_NAME
    if 'cantidad_arcos_widget' not in st.session_state: # Corrected key
        st.session_state.cantidad_arcos_widget = 1 # Corrected key

    # Usar los mismos valores por defecto que el resto de la app
    if 'chord_input_float' not in st.session_state:
        st.session_state.chord_input_float = 1000.0
    if 'sagitta_input_float' not in st.session_state:
        st.session_state.sagitta_input_float = 250.0
    if 'tube_length_input_float' not in st.session_state:
        if st.session_state.selected_unit_name == "Centímetros (cm)":
            st.session_state.tube_length_input_float = 600.0
        else:
            current_unit_factor = UNITS_TO_METERS[st.session_state.selected_unit_name]
            st.session_state.tube_length_input_float = float(600.0 * float(UNITS_TO_METERS["Centímetros (cm)"]) / float(current_unit_factor))
    if 'tam_regla_input' not in st.session_state:
        if st.session_state.selected_unit_name == "Centímetros (cm)":
            st.session_state.tam_regla_input = 216.0
        else:
            current_unit_factor = UNITS_TO_METERS[st.session_state.selected_unit_name]
            st.session_state.tam_regla_input = float(216.0 * float(UNITS_TO_METERS["Centímetros (cm)"]) / float(current_unit_factor))
    if 'anc_regla_input' not in st.session_state:
        if st.session_state.selected_unit_name == "Centímetros (cm)":
            st.session_state.anc_regla_input = 3.781
        else:
            current_unit_factor = UNITS_TO_METERS[st.session_state.selected_unit_name]
            st.session_state.anc_regla_input = float(3.781 * float(UNITS_TO_METERS["Centímetros (cm)"]) / float(current_unit_factor))

    main()


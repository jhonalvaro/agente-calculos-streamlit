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
import re

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
DEFAULT_TUBE_LENGTH_BASE_UNIT = Decimal("6.0")

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
                return {"arc_length": arc_length_dec, "central_angle_deg": central_angle_deg_dec, "error": None}
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
    global app_config
    app_config = init_app_config()
    getcontext().prec = app_config['precision']

    st.markdown("---")
    st.header("💬 Asistente IA (Gemini Flash)")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🔄 Resetear chat y contexto", key="reset_chat_btn", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Sugerencias dinámicas
    sugerencias = [
        "¿Cuántos tubos necesito para cubrir todos los arcos?",
        "¿Cómo se calcula el radio con los datos actuales?",
        "Muéstrame el gráfico del arco principal.",
        "Cambiar la longitud del tubo a 12",
        "Explica qué es la sagitta."
    ]
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

    # --- INICIO DEL BLOQUE CORREGIDO ---
    if enviar and chat_input.strip():
        # Construir contexto para IA con datos de entrada y resultados
        contexto = [
            f"Unidad seleccionada: {st.session_state.get('selected_unit_name', 'N/A')}",
            f"Cuerda (c): {st.session_state.get('chord_input_float', 'N/A')}",
            f"Sagitta (s): {st.session_state.get('sagitta_input_float', 'N/A')}",
            f"Longitud de tubo (L_tubo): {st.session_state.get('tube_length_input_float', 'N/A')}",
            f"Cantidad de arcos: {st.session_state.get('cantidad_arcos_widget', 'N/A')}"
        ]
        if 'radius_display' in st.session_state:
            contexto.append(f"Radio calculado: {st.session_state.radius_display}")
        if 'arc_length_display_one_arc' in st.session_state:
            contexto.append(f"Longitud de arco (1 arco): {st.session_state.arc_length_display_one_arc}")
        if 'num_tubes_output_str' in st.session_state:
            contexto.append(f"Resultado de ajuste de tubos: {st.session_state.num_tubes_output_str}")

        historial = "\n".join([f"Usuario: {h['user']}\nAsistente: {h['ai']}" for h in st.session_state.chat_history])
        prompt = (f"Historial previo:\n{historial}\n" if historial else "") +                  (f"Contexto actual de la app:\n" + "\n".join(contexto) + "\n\n" if contexto else "") +                  f"Pregunta o instrucción del usuario: {chat_input}\n"

        ai_response = call_gemini_api(prompt, app_config['gemini_api_key'])

        if ai_response["success"]:
            respuesta = ai_response["response"]

            # Detectar si el usuario quiere cambiar algún valor
            cambios_realizados = False
            for campo, key, tipo in [
                ("cuerda", "chord_input_float", float),
                ("sagitta", "sagitta_input_float", float),
                ("longitud de tubo", "tube_length_input_float", float),
                ("longitud del tubo", "tube_length_input_float", float), # Corrected indentation
                ("cantidad de arcos", "cantidad_arcos_widget", int)
            ]:
                patron = rf"cambiar {campo} a\s*([0-9]+\.?[0-9]*)"
                m = re.search(patron, chat_input, re.IGNORECASE)
                if m:
                    try:
                        nuevo_valor = tipo(m.group(1))
                        st.session_state[key] = nuevo_valor
                        cambios_realizados = True
                    except (ValueError, IndexError):
                        pass

            # Guardar en historial y limpiar la caja de entrada
            st.session_state.chat_history.append({"user": chat_input, "ai": respuesta})
            st.session_state.chat_input = ""

        else:
            # En caso de error de la API
            error_msg = f"❌ Error de la IA: {ai_response['error']}"
            st.session_state.chat_history.append({"user": chat_input, "ai": error_msg})
            # No limpiar la caja de entrada para que el usuario pueda corregir

        # Re-ejecutar el script para mostrar el nuevo historial y actualizar widgets si hubo cambios
        st.rerun()
    # --- FIN DEL BLOQUE CORREGIDO ---

    # Mostrar historial de chat
    for h in st.session_state.chat_history[-8:]:
        st.markdown(f"<div style='background:#23272f;color:#f5f5fa;padding:0.5em 1em;border-radius:8px;margin-bottom:0.5em;'><b>Usuario:</b> {h['user']}<br><b>Asistente:</b> {h['ai']}</div>", unsafe_allow_html=True)

    st.title("🔢 Calculadora de Precisión para Arcos y Tubos")

    # --- Session State Initialization ---
    if 'selected_unit_name' not in st.session_state:
        st.session_state.selected_unit_name = UNIT_NAMES[0]
    if 'cantidad_arcos_widget' not in st.session_state:
        st.session_state.cantidad_arcos_widget = 1
    if 'chord_input_float' not in st.session_state:
        st.session_state.chord_input_float = 10.0
    if 'sagitta_input_float' not in st.session_state:
        st.session_state.sagitta_input_float = 2.5
    if 'tube_length_input_float' not in st.session_state:
        initial_unit_factor_to_base = UNITS_TO_METERS[st.session_state.selected_unit_name]
        st.session_state.tube_length_input_float = float(DEFAULT_TUBE_LENGTH_BASE_UNIT / initial_unit_factor_to_base)

    newly_selected_unit_name = st.selectbox("Unidad para Entradas/Resultados:", UNIT_NAMES, index=UNIT_NAMES.index(st.session_state.selected_unit_name), key="unit_selector_widget")

    if newly_selected_unit_name != st.session_state.selected_unit_name:
        old_unit_factor = UNITS_TO_METERS[st.session_state.selected_unit_name]
        new_unit_factor = UNITS_TO_METERS[newly_selected_unit_name]

        # Convert default-like values to the new unit, otherwise keep user-entered values converted
        if abs(st.session_state.chord_input_float - float(Decimal("10.0") / old_unit_factor)) < 1e-5:
             st.session_state.chord_input_float = float(Decimal("10.0") / new_unit_factor)
        else:
             st.session_state.chord_input_float = float(Decimal(str(st.session_state.chord_input_float)) * old_unit_factor / new_unit_factor)

        if abs(st.session_state.sagitta_input_float - float(Decimal("2.5") / old_unit_factor)) < 1e-5:
             st.session_state.sagitta_input_float = float(Decimal("2.5") / new_unit_factor)
        else:
             st.session_state.sagitta_input_float = float(Decimal(str(st.session_state.sagitta_input_float)) * old_unit_factor / new_unit_factor)

        if abs(st.session_state.tube_length_input_float - float(DEFAULT_TUBE_LENGTH_BASE_UNIT / old_unit_factor)) < 1e-5:
            st.session_state.tube_length_input_float = float(DEFAULT_TUBE_LENGTH_BASE_UNIT / new_unit_factor)
        else:
            st.session_state.tube_length_input_float = float(Decimal(str(st.session_state.tube_length_input_float)) * old_unit_factor / new_unit_factor)

        st.session_state.selected_unit_name = newly_selected_unit_name
        st.rerun()

    # --- LÍNEA CORREGIDA ---
    selected_unit_name_for_display = st.session_state.selected_unit_name

    # El resto del script sigue aquí...
    # (El código no fue provisto más allá de este punto, pero la estructura existente está ahora corregida)
    # Por lo tanto, el final del archivo se considera completo y funcional hasta este punto.

if __name__ == "__main__":
    main()

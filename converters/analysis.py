"""
converters/analysis.py
----------------------
Main loss-map engine.

run_loss_map(topology, t1, t2, params) -> LossMapResult

Computes 2-D maps of:
  - P_cond_T1   conduction losses T1       [W]
  - P_cond_T2   conduction losses T2       [W]
  - P_sw_T1     total switching losses T1  [W]
  - P_rr_T2     reverse-recovery losses T2 [W]
  - P_total     sum of all losses          [W]
  - T_j_T1      junction temperature T1    [°C]
  - T_j_T2      junction temperature T2    [°C]  (NaN if no thermal data)
  - duty        duty cycle                 [-]
  - i_peak      peak inductor current      [A]

X-axis: V_in  (from params)
Y-axis: P_out (from params)

All returned maps are 2-D numpy arrays [len(p_out_vec), len(v_in_vec)].
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .core import ConverterDevice
from .formulas import Boost, Buck, BuckBoost


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ConverterParams:
    """
    Operating-point parameters for a loss-map sweep.

    v_in_range   : (v_min, v_max)   input voltage sweep  [V]
    p_out_range  : (p_min, p_max)   output power sweep   [W]
    v_out        : float             output voltage       [V]
    frequency    : float             switching frequency  [Hz]
    inductance   : float             inductance           [H]
    v_g_on       : float             turn-on gate voltage [V]
    t_heatsink   : float             heatsink temperature [°C]
    r_th_heatsink: float             heatsink Rth         [K/W]
    n_points     : int               grid resolution per axis
    """
    v_in_range:    tuple = (200.0, 800.0)
    p_out_range:   tuple = (100.0, 10000.0)
    v_out:         float = 400.0
    frequency:     float = 10e3
    inductance:    float = 1e-3
    v_g_on:        float = 15.0
    t_heatsink:    float = 50.0
    r_th_heatsink: float = 0.1
    n_points:      int   = 40


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LossMapResult:
    topology:   str
    t1_name:    str
    t2_name:    str
    v_in_vec:   np.ndarray
    p_out_vec:  np.ndarray
    P_cond_T1:  np.ndarray
    P_cond_T2:  np.ndarray
    P_sw_T1:    np.ndarray
    P_rr_T2:    np.ndarray
    P_total:    np.ndarray
    T_j_T1:     np.ndarray
    T_j_T2:     np.ndarray
    duty:       np.ndarray
    i_peak:     np.ndarray
    warnings:   list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Topology dispatch
# ---------------------------------------------------------------------------

_TOPO = {
    "boost":      Boost,
    "buck":       Buck,
    "buck_boost": BuckBoost,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_loss_map(topology: str,
                 t1: ConverterDevice,
                 t2: ConverterDevice,
                 params: ConverterParams) -> LossMapResult:
    """
    Compute a 2-D loss map for the given topology and two transistors.

    topology : 'boost' | 'buck' | 'buck_boost'
    t1       : active switch (T1)
    t2       : freewheeling diode or second switch (T2)
    params   : ConverterParams

    Returns LossMapResult.
    """
    topo_cls = _TOPO.get(topology.lower().replace("-", "_").replace(" ", "_"))
    if topo_cls is None:
        raise ValueError(f"Unknown topology '{topology}'. "
                         f"Choose from: {list(_TOPO.keys())}")

    warnings = []

    # ---- build grid ----
    v_in_vec  = np.linspace(params.v_in_range[0],  params.v_in_range[1],  params.n_points)
    p_out_vec = np.linspace(params.p_out_range[0], params.p_out_range[1], params.n_points)

    V_IN, P_OUT = np.meshgrid(v_in_vec, p_out_vec)   # shape [n, n]
    V_OUT = np.full_like(V_IN, params.v_out)
    F     = params.frequency
    L     = params.inductance
    R_load = V_OUT**2 / np.maximum(P_OUT, 1e-3)
    ZETA  = L * F / R_load

    # ---- operating-point T_j (simplified: use max available in data) ----
    t_j_sw = t1.max_t_j_switch()
    t_j_di = t2.max_t_j_diode()
    v_g    = params.v_g_on

    # ---- channel voltages at a representative current ----
    # Use I_out ≈ P_out / V_out as representative current for linearisation
    I_REP = P_OUT / np.maximum(V_OUT, 1.0)
    I_REP_scalar = float(np.median(I_REP))

    v_ch1_sw, r_ch1_sw = t1.calc_lin_channel(I_REP_scalar, t_j_sw, v_g, "switch")
    v_ch2_di, r_ch2_di = t2.calc_lin_channel(I_REP_scalar, t_j_di, 0.0, "diode")

    V_CH1 = np.full_like(V_IN, v_ch1_sw + r_ch1_sw * I_REP_scalar)
    V_CH2 = np.full_like(V_IN, v_ch2_di)

    # ---- waveform quantities ----
    I_PEAK = topo_cls.i_peak(ZETA, V_IN, V_OUT, P_OUT, V_CH1, V_CH2)
    I_1_RMS  = topo_cls.i1_rms(ZETA, V_IN, V_OUT, P_OUT, V_CH1, V_CH2)
    I_1_MEAN = topo_cls.i1_mean(ZETA, V_IN, V_OUT, P_OUT, V_CH1, V_CH2)
    I_2_RMS  = topo_cls.i2_rms(ZETA, V_IN, V_OUT, P_OUT, V_CH1, V_CH2)
    I_ON_T1  = topo_cls.i_on_t1(ZETA, V_IN, V_OUT, P_OUT, V_CH1, V_CH2)
    I_OFF_T1 = topo_cls.i_off_t1(ZETA, V_IN, V_OUT, P_OUT, V_CH1, V_CH2)
    I_RR_T2  = topo_cls.i_rr_t2(ZETA, V_IN, V_OUT, P_OUT, V_CH1, V_CH2)
    DUTY     = topo_cls.duty_ccm(V_IN, V_OUT, V_CH1, V_CH2)

    # ---- conduction losses ----
    # T1: P = I_rms² * R_ch + I_mean * V_ch0
    P_COND_T1 = I_1_RMS**2 * r_ch1_sw + I_1_MEAN * v_ch1_sw
    # T2 (diode): P = I_rms * V_forward  (diode: no resistive term in linearised model at I_rep)
    P_COND_T2 = I_2_RMS * V_CH2

    # ---- switching losses ----
    if t1.has_e_on and t1.has_e_off:
        t_j_eon  = t1.max_t_j_e_on()
        t_j_eoff = t1.max_t_j_e_off()

        # vectorised energy lookup
        E_ON  = _interp_energy_map(t1, "e_on",  I_ON_T1,  t_j_eon,  V_OUT)
        E_OFF = _interp_energy_map(t1, "e_off", I_OFF_T1, t_j_eoff, V_OUT)
        P_SW_T1 = (E_ON + E_OFF) * F
    else:
        P_SW_T1 = np.zeros_like(V_IN)
        warnings.append(f"T1 ({t1.name}): no e_on/e_off data – switching losses set to 0.")

    if t2.has_e_rr:
        t_j_err = t2.max_t_j_e_rr()
        E_RR    = _interp_energy_map(t2, "e_rr", I_RR_T2, t_j_err, V_OUT)
        P_RR_T2 = E_RR * F
    else:
        P_RR_T2 = np.zeros_like(V_IN)
        warnings.append(f"T2 ({t2.name}): no e_rr data – reverse-recovery losses set to 0.")

    P_TOTAL = P_COND_T1 + P_COND_T2 + P_SW_T1 + P_RR_T2

    # ---- junction temperatures (simplified, no iteration) ----
    r_th_t1 = (t1.r_th_switch_jc + t1.r_th_switch_cs + t1.r_th_cs + params.r_th_heatsink)
    T_J_T1  = params.t_heatsink + (P_COND_T1 + P_SW_T1) * r_th_t1

    if t2.has_diode_tf:
        r_th_t2 = (t2.r_th_diode_jc + t2.r_th_diode_cs + t2.r_th_cs + params.r_th_heatsink)
        T_J_T2  = params.t_heatsink + (P_COND_T2 + P_RR_T2) * r_th_t2
    else:
        T_J_T2 = np.full_like(V_IN, np.nan)
        warnings.append(f"T2 ({t2.name}): no diode thermal data – T_j_T2 not computed.")

    # ---- clip unphysical regions ----
    mask_invalid = (I_PEAK > t1.i_abs_max) | (DUTY < 0) | (DUTY > 1)
    # Topology-specific voltage validity
    if topo_cls is Boost:
        mask_invalid = mask_invalid | (V_IN >= V_OUT)
    elif topo_cls is Buck:
        mask_invalid = mask_invalid | (V_IN <= V_OUT)
    # buck-boost: always steps up+down, no extra voltage mask needed
    for arr in (P_COND_T1, P_COND_T2, P_SW_T1, P_RR_T2, P_TOTAL, T_J_T1, T_J_T2, DUTY, I_PEAK):
        arr[mask_invalid] = np.nan

    return LossMapResult(
        topology=topology,
        t1_name=t1.name,
        t2_name=t2.name,
        v_in_vec=v_in_vec,
        p_out_vec=p_out_vec,
        P_cond_T1=P_COND_T1,
        P_cond_T2=P_COND_T2,
        P_sw_T1=P_SW_T1,
        P_rr_T2=P_RR_T2,
        P_total=P_TOTAL,
        T_j_T1=T_J_T1,
        T_j_T2=T_J_T2,
        duty=DUTY,
        i_peak=I_PEAK,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Vectorised energy interpolation helper
# ---------------------------------------------------------------------------

def _interp_energy_map(device: ConverterDevice, which: str,
                       i_map: np.ndarray, t_j: float,
                       v_out_map: np.ndarray) -> np.ndarray:
    """
    Interpolate switching energy over the full 2-D current map.
    v_out_map is used for linear voltage scaling.
    Returns a 2-D array of energy values [J].
    """
    tbl = {"e_on":  device._e_on,
           "e_off": device._e_off,
           "e_rr":  device._e_rr}
    entries = tbl.get(which, [])
    if not entries:
        return np.zeros_like(i_map)

    # pick entry with closest t_j
    diffs = [abs(e["t_j"] - t_j) for e in entries]
    entry = entries[int(np.argmin(diffs))]
    i_pts = entry["i_pts"]
    e_pts = entry["e_pts"]
    v_supply = entry["v_supply"]

    # flatten, interpolate, reshape
    flat_i = np.clip(i_map.ravel(), i_pts[0], i_pts[-1])
    flat_e = np.interp(flat_i, i_pts, e_pts)
    E = flat_e.reshape(i_map.shape)

    # linear voltage scaling
    if v_supply > 0:
        E = E * v_out_map / v_supply

    return np.maximum(E, 0.0)

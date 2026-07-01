"""
converters/gui_tab.py
---------------------
Self-contained ttk.Frame that plugs into the main Notebook as a
'⚡ Converters' tab.

Usage in GUI.py:
    from converters.gui_tab import ConverterTab
    tab = ttk.Frame(self.nb)
    self.nb.add(tab, text="⚡ Converters")
    ConverterTab(tab, df=self.df)
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from .core import ConverterDevice, ConverterError
from .analysis import ConverterParams, LossMapResult, run_loss_map

# ---------------------------------------------------------------------------
# Colour palette (matches existing GUI style)
# ---------------------------------------------------------------------------
CLR_BG      = "#f7f7f7"
CLR_ACCENT  = "#2980b9"
CLR_GREEN   = "#27ae60"
CLR_RED     = "#e74c3c"
CLR_WARN    = "#e67e22"

TOPOLOGIES = ["boost", "buck", "buck_boost"]
TOPO_LABELS = {"boost": "Boost", "buck": "Buck", "buck_boost": "Buck-Boost"}

MAP_CHOICES = [
    ("P_total",   "Total losses [W]"),
    ("P_cond_T1", "T1 Conduction losses [W]"),
    ("P_cond_T2", "T2 Conduction losses [W]"),
    ("P_sw_T1",   "T1 Switching losses [W]"),
    ("P_rr_T2",   "T2 Reverse-recovery [W]"),
    ("T_j_T1",    "T1 Junction temp [°C]"),
    ("T_j_T2",    "T2 Junction temp [°C]"),
    ("duty",      "Duty cycle [-]"),
    ("i_peak",    "Peak current [A]"),
]


class ConverterTab:
    """
    Builds and owns the Converter loss-map UI inside parent_frame.
    Call ConverterTab(parent_frame, df=dataframe) once.
    """

    def __init__(self, parent: tk.Widget, df=None):
        self.parent = parent
        self.df = df              # pandas DataFrame from load_full_database()
        self._result: LossMapResult | None = None
        self._t1: ConverterDevice | None = None
        self._t2: ConverterDevice | None = None

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # ---- main paned window: left=controls, right=plot ----
        pw = ttk.PanedWindow(parent, orient="horizontal")
        pw.grid(row=0, column=0, sticky="nsew")

        left  = ttk.Frame(pw, width=380)
        right = ttk.Frame(pw)
        left.pack_propagate(False); left.grid_propagate(False)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        self._build_controls(left)
        self._build_plot_area(right)

    # ------------------------------------------------------------------
    # Controls panel
    # ------------------------------------------------------------------
    def _build_controls(self, p: tk.Widget):
        p.columnconfigure(0, weight=1)

        ttk.Label(p, text="⚡ Converter Loss Map",
                  font=("Arial", 13, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 4))

        # ---- topology ----
        tf = ttk.LabelFrame(p, text=" Topology ", padding=8)
        tf.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        tf.columnconfigure(1, weight=1)
        self._topo_var = tk.StringVar(value="boost")
        for i, topo in enumerate(TOPOLOGIES):
            ttk.Radiobutton(tf, text=TOPO_LABELS[topo],
                            variable=self._topo_var, value=topo).grid(
                row=0, column=i, padx=8, sticky="w")

        # ---- device selectors ----
        df_frame = ttk.LabelFrame(p, text=" Device Selection ", padding=8)
        df_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        df_frame.columnconfigure(1, weight=1)

        for row_idx, (label, var_attr, cb_attr) in enumerate([
                ("T1 – Switch:",        "_t1_var", "_t1_combo"),
                ("T2 – Diode / Switch:","_t2_var", "_t2_combo")]):
            ttk.Label(df_frame, text=label, font=("Arial", 9, "bold")).grid(
                row=row_idx, column=0, sticky="w", padx=4, pady=4)
            var = tk.StringVar()
            setattr(self, var_attr, var)
            cb = ttk.Combobox(df_frame, textvariable=var,
                              state="readonly", font=("Consolas", 8), width=32)
            cb.grid(row=row_idx, column=1, sticky="ew", padx=4, pady=4)
            setattr(self, cb_attr, cb)

        self._refresh_device_lists()

        # ---- parameters ----
        pf = ttk.LabelFrame(p, text=" Operating Parameters ", padding=8)
        pf.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        pf.columnconfigure(1, weight=1); pf.columnconfigure(3, weight=1)

        param_defs = [
            # (label,          attr_suffix,    default,   unit,   col)
            ("V_out [V]",      "v_out",        400.0,     "",     0),
            ("V_in min [V]",   "v_in_min",     200.0,     "",     0),
            ("V_in max [V]",   "v_in_max",     800.0,     "",     0),
            ("P_out min [W]",  "p_out_min",    500.0,     "",     0),
            ("P_out max [W]",  "p_out_max",    10000.0,   "",     0),
            ("Frequency [Hz]", "frequency",    10000.0,   "",     2),
            ("Inductance [H]", "inductance",   1e-3,      "",     2),
            ("V_g_on [V]",     "v_g_on",       15.0,      "",     2),
            ("T_heatsink [°C]","t_heatsink",   50.0,      "",     2),
            ("Rth_sink [K/W]", "r_th_heatsink",0.1,       "",     2),
            ("Grid points",    "n_points",     40,        "",     2),
        ]
        self._param_vars = {}
        rows_left, rows_right = 0, 0
        for label, key, default, unit, col in param_defs:
            r = rows_left if col == 0 else rows_right
            ttk.Label(pf, text=label, font=("Arial", 8)).grid(
                row=r, column=col, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(default))
            ttk.Entry(pf, textvariable=var, width=10,
                      font=("Consolas", 9)).grid(
                row=r, column=col+1, sticky="ew", padx=4, pady=2)
            self._param_vars[key] = var
            if col == 0: rows_left += 1
            else: rows_right += 1

        # ---- map selector ----
        mf = ttk.LabelFrame(p, text=" Map to Display ", padding=8)
        mf.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        mf.columnconfigure(1, weight=1)
        self._map_var = tk.StringVar(value="P_total")
        self._map_combo = ttk.Combobox(
            mf, textvariable=self._map_var,
            values=[k for k, _ in MAP_CHOICES],
            state="readonly", font=("Consolas", 9))
        self._map_combo.grid(row=0, column=0, sticky="ew", padx=4)
        self._map_combo.set("P_total")
        self._map_combo.bind("<<ComboboxSelected>>", lambda e: self._replot())

        # ---- run button ----
        bf = ttk.Frame(p, padding=(0, 4))
        bf.grid(row=5, column=0, sticky="ew", padx=8, pady=6)
        self._run_btn = ttk.Button(bf, text="▶  Run Loss Map",
                                   command=self._run)
        self._run_btn.pack(side="left", ipady=6, ipadx=10)
        self._status_lbl = ttk.Label(bf, text="", foreground="gray",
                                     font=("Arial", 9))
        self._status_lbl.pack(side="left", padx=10)

        # ---- warnings box ----
        wf = ttk.LabelFrame(p, text=" Warnings ", padding=6)
        wf.grid(row=6, column=0, sticky="ew", padx=8, pady=4)
        wf.columnconfigure(0, weight=1)
        self._warn_text = tk.Text(wf, height=4, font=("Arial", 8),
                                  wrap="word", state="disabled",
                                  background="#fff8e1")
        self._warn_text.grid(row=0, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # Plot area
    # ------------------------------------------------------------------
    def _build_plot_area(self, p: tk.Widget):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(0, weight=1)

        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

            self._fig = Figure(figsize=(7, 5), dpi=96,
                               facecolor=CLR_BG, tight_layout=True)
            self._ax  = self._fig.add_subplot(111)
            self._ax.set_facecolor("#eef2f7")
            self._ax.text(0.5, 0.5, "Select devices and click ▶  Run Loss Map",
                          ha="center", va="center", fontsize=12,
                          color="gray", transform=self._ax.transAxes)

            canvas = FigureCanvasTkAgg(self._fig, master=p)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

            tb_frame = ttk.Frame(p)
            tb_frame.grid(row=1, column=0, sticky="ew")
            NavigationToolbar2Tk(canvas, tb_frame)

            self._canvas = canvas

        except ImportError:
            ttk.Label(p, text="matplotlib not installed.\nRun: pip install matplotlib",
                      foreground=CLR_RED, font=("Arial", 12)).grid(
                row=0, column=0, padx=20, pady=20)
            self._canvas = None

    # ------------------------------------------------------------------
    # Device list refresh
    # ------------------------------------------------------------------
    def _refresh_device_lists(self):
        if self.df is None or self.df.empty:
            return
        names = sorted(self.df["name"].dropna().tolist())
        for attr in ("_t1_combo", "_t2_combo"):
            cb = getattr(self, attr, None)
            if cb is not None:
                cb["values"] = names

    def update_df(self, df):
        """Call this when the database is reloaded from GUI.py."""
        self.df = df
        self._refresh_device_lists()

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------
    def _parse_params(self) -> ConverterParams | None:
        try:
            return ConverterParams(
                v_in_range=(float(self._param_vars["v_in_min"].get()),
                            float(self._param_vars["v_in_max"].get())),
                p_out_range=(float(self._param_vars["p_out_min"].get()),
                             float(self._param_vars["p_out_max"].get())),
                v_out=float(self._param_vars["v_out"].get()),
                frequency=float(self._param_vars["frequency"].get()),
                inductance=float(self._param_vars["inductance"].get()),
                v_g_on=float(self._param_vars["v_g_on"].get()),
                t_heatsink=float(self._param_vars["t_heatsink"].get()),
                r_th_heatsink=float(self._param_vars["r_th_heatsink"].get()),
                n_points=int(self._param_vars["n_points"].get()),
            )
        except ValueError as e:
            messagebox.showerror("Parameter Error",
                                 f"Invalid value in parameters:\n{e}")
            return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _run(self):
        t1_name = self._t1_var.get()
        t2_name = self._t2_var.get()
        if not t1_name or not t2_name:
            messagebox.showwarning("Selection",
                                   "Select both T1 (switch) and T2 (diode/switch).")
            return

        params = self._parse_params()
        if params is None:
            return

        topo = self._topo_var.get()

        # Load ConverterDevice objects
        def _get_path(name):
            row = self.df[self.df["name"] == name]
            if row.empty:
                raise ConverterError(f"Device '{name}' not found in database.")
            return row.iloc[0]["_original_file_path"]

        try:
            t1_path = _get_path(t1_name)
            t2_path = _get_path(t2_name)
            t1 = ConverterDevice(t1_path)
            t2 = ConverterDevice(t2_path)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        self._run_btn.config(state="disabled")
        self._status_lbl.config(text="⏳ Computing…", foreground=CLR_WARN)
        self.parent.update_idletasks()

        def _worker():
            try:
                result = run_loss_map(topo, t1, t2, params)
                self.parent.after(0, lambda: self._on_result(result))
            except Exception as e:
                self.parent.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_result(self, result: LossMapResult):
        self._result = result
        self._run_btn.config(state="normal")

        # warnings
        self._warn_text.config(state="normal")
        self._warn_text.delete("1.0", "end")
        if result.warnings:
            for w in result.warnings:
                self._warn_text.insert("end", f"⚠ {w}\n")
        else:
            self._warn_text.insert("end", "No warnings.")
        self._warn_text.config(state="disabled")

        self._status_lbl.config(
            text=f"✅ Done  ({result.topology}  {result.t1_name} / {result.t2_name})",
            foreground=CLR_GREEN)
        self._replot()

    def _on_error(self, msg: str):
        self._run_btn.config(state="normal")
        self._status_lbl.config(text="⚠ Error", foreground=CLR_RED)
        messagebox.showerror("Computation Error", msg)

    # ------------------------------------------------------------------
    # Plot rendering
    # ------------------------------------------------------------------
    def _replot(self):
        if self._result is None or self._canvas is None:
            return

        result  = self._result
        map_key = self._map_var.get()
        data    = getattr(result, map_key, None)
        if data is None:
            return

        label = next((lbl for k, lbl in MAP_CHOICES if k == map_key), map_key)

        self._ax.cla()

        # build meshgrid for pcolormesh
        V, P = np.meshgrid(result.v_in_vec, result.p_out_vec)
        masked = np.ma.masked_invalid(data)

        pc = self._ax.pcolormesh(V, P, masked, shading="auto",
                                 cmap="plasma")

        # colorbar – remove previous if any
        if hasattr(self, "_cbar") and self._cbar is not None:
            try: self._cbar.remove()
            except Exception: pass
        self._cbar = self._fig.colorbar(pc, ax=self._ax, fraction=0.046, pad=0.04)
        self._cbar.set_label(label, fontsize=9)

        # contour overlay for total losses
        if map_key in ("P_total", "P_cond_T1", "P_cond_T2", "P_sw_T1"):
            try:
                levels = np.linspace(np.nanmin(data), np.nanmax(data), 8)
                cs = self._ax.contour(V, P, masked, levels=levels,
                                      colors="white", linewidths=0.6, alpha=0.5)
                self._ax.clabel(cs, fmt="%.0f W", fontsize=7, inline=True)
            except Exception:
                pass

        # axis labels
        self._ax.set_xlabel("Input Voltage V_in [V]", fontsize=9)
        self._ax.set_ylabel("Output Power P_out [W]", fontsize=9)

        topo_label = TOPO_LABELS.get(result.topology, result.topology)
        self._ax.set_title(
            f"{topo_label}  –  {label}\n"
            f"T1: {result.t1_name}    T2: {result.t2_name}",
            fontsize=9)

        self._canvas.draw()
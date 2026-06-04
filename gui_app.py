#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

from equilibrium import find_thermo_db
from nasa9_parser import parse_thermo_file
from nozzle_flow import optimize_lox_lh2_mixture_ratio, get_valid_propellant_components


PC_DEFAULT_MPA = 10.0
PE_DEFAULT_ATM = 1.0
PE_DEFAULT_PA = 101325.0


def _top_species_indices(stations, top_k=8):
    if not stations:
        return []
    n = len(stations[0].species_names)
    max_x = [0.0] * n
    for st in stations:
        for i, x in enumerate(st.mole_fractions):
            if x > max_x[i]:
                max_x[i] = float(x)
    order = sorted(range(n), key=lambda i: max_x[i], reverse=True)
    return [i for i in order[:top_k] if max_x[i] > 1e-10]


def _format_result_text(opt_result: dict) -> str:
    perf = opt_result["perf"]
    stations = perf.stations  # строго 4: Injector / Nozzle inlet / Nozzle throat / Nozzle exit

    lines = []
    lines.append("Оптимизация завершена")
    lines.append("=" * 80)
    lines.append(f"Pc = {PC_DEFAULT_MPA:.4f} МПа")
    lines.append(f"Pe = {PE_DEFAULT_ATM:.4f} атм ({PE_DEFAULT_PA/1e6:.6f} МПа)")
    lines.append("Топливо: LOX/LH2 (O2(L) + H2(L))")
    lines.append("Сечения: вход в КС, начало сопла, критика, срез")
    lines.append("")

    lines.append(f"alpha_opt           = {opt_result['alpha_opt']:.6f}")
    lines.append(f"O/F_stoich          = {opt_result['of_stoich']:.6f}")
    lines.append(f"O/F_opt             = {opt_result['of_opt']:.6f}")
    lines.append(f"Isp_opt             = {opt_result['isp_opt_s']:.4f} c")
    lines.append(f"Isp_vac             = {perf.Isp_vac_s:.4f} c")
    lines.append(f"C*                  = {perf.Cstar_m_per_s:.4f} м/с")
    lines.append(f"CF                  = {perf.CF:.4f}")
    lines.append(f"Оценок целевой ф-ции (ускоренный поиск) = {opt_result['evaluations']}")
    lines.append("")

    lines.append("Термодинамика по 4 сечениям")
    lines.append("-" * 80)
    lines.append(f"{'Сечение':<16} {'P,MPa':>10} {'T,K':>11} {'M':>9} {'V,m/s':>12} {'Ae/At':>11}")
    for st in stations:
        lines.append(
            f"{st.label:<16} {st.P_Pa/1e6:>10.5f} {st.T_K:>11.3f} {st.M:>9.4f} {st.V_m_per_s:>12.3f} {st.Ae_At:>11.4f}"
        )

    top_idx = _top_species_indices(stations, top_k=8)
    if top_idx:
        lines.append("")
        lines.append("Fractions of combustion products (mole / mass)")
        lines.append("-" * 80)
        hdr = f"{'Species':<14}"
        for st in stations:
            hdr += f" {st.label + ' x':>14} {st.label + ' w':>14}"
        lines.append(hdr)

        sp_names = stations[0].species_names
        for idx in top_idx:
            row = f"{sp_names[idx]:<14}"
            for st in stations:
                row += f" {float(st.mole_fractions[idx]):>14.7f} {float(st.mass_fractions[idx]):>14.7f}"
            lines.append(row)

    lines.append("")
    return "\n".join(lines)


class RocketEquilibriumGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LOX/LH2 Equilibrium Nozzle (GUI)")
        self.root.geometry("1280x760")

        self.species_db = None

        self._build_ui()
        self._load_db()

    def _build_ui(self):
        frm_top = ttk.Frame(self.root, padding=10)
        frm_top.pack(fill=tk.X)

        ttk.Label(frm_top, text="Режим: только GUI, только LOX/LH2", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))

        ttk.Label(frm_top, text="Окислитель:").grid(row=1, column=0, sticky="w")
        self.var_ox = tk.StringVar(value="O2(L)")
        ttk.Entry(frm_top, textvariable=self.var_ox, width=16, state="readonly").grid(row=1, column=1, sticky="w", padx=(6, 18))

        ttk.Label(frm_top, text="Горючее:").grid(row=1, column=2, sticky="w")
        self.var_fu = tk.StringVar(value="H2(L)")
        ttk.Entry(frm_top, textvariable=self.var_fu, width=16, state="readonly").grid(row=1, column=3, sticky="w", padx=(6, 18))

        ttk.Label(frm_top, text="Pc, МПа:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.var_pc = tk.StringVar(value=f"{PC_DEFAULT_MPA:g}")
        ttk.Entry(frm_top, textvariable=self.var_pc, width=16, state="readonly").grid(row=2, column=1, sticky="w", padx=(6, 18), pady=(6, 0))

        ttk.Label(frm_top, text="Pe, атм:").grid(row=2, column=2, sticky="w", pady=(6, 0))
        self.var_pe = tk.StringVar(value=f"{PE_DEFAULT_ATM:g}")
        ttk.Entry(frm_top, textvariable=self.var_pe, width=16, state="readonly").grid(row=2, column=3, sticky="w", padx=(6, 18), pady=(6, 0))

        ttk.Label(frm_top, text="Сечения:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.var_sections = tk.StringVar(value="4 (вход КС, начало сопла, критика, срез)")
        ttk.Entry(frm_top, textvariable=self.var_sections, width=52, state="readonly").grid(row=3, column=1, columnspan=3, sticky="w", padx=(6, 18), pady=(6, 0))

        self.btn_run = ttk.Button(frm_top, text="Найти оптимальное O/F и рассчитать", command=self.start_calculation)
        self.btn_run.grid(row=1, column=4, rowspan=2, sticky="nsew", padx=(14, 0))

        self.lbl_status = ttk.Label(frm_top, text="Готово.")
        self.lbl_status.grid(row=3, column=4, sticky="w", padx=(14, 0), pady=(6, 0))

        frm_text = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        frm_text.pack(fill=tk.BOTH, expand=True)

        self.txt = ScrolledText(frm_text, wrap=tk.NONE, font=("Consolas", 10))
        self.txt.pack(fill=tk.BOTH, expand=True)

    def _load_db(self):
        try:
            db_path = find_thermo_db()
            self.species_db = parse_thermo_file(db_path)
            oxidizers, fuels = get_valid_propellant_components(self.species_db)
            self.lbl_status.config(
                text=f"База загружена: {len(self.species_db)} веществ. Допустимых окислителей: {len(oxidizers)}, горючих: {len(fuels)} (ионы исключены)."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось загрузить NASA-базу: {exc}")
            self.btn_run.config(state=tk.DISABLED)

    def start_calculation(self):
        if self.species_db is None:
            messagebox.showerror("Ошибка", "NASA-база не загружена")
            return

        self.btn_run.config(state=tk.DISABLED)
        self.lbl_status.config(text="Расчёт выполняется...")
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "Выполняется ускоренный поиск оптимального соотношения...\n")

        thread = threading.Thread(target=self._run_calculation_worker, daemon=True)
        thread.start()

    def _run_calculation_worker(self):
        try:
            result = optimize_lox_lh2_mixture_ratio(
                species_db=self.species_db,
                P_chamber=PC_DEFAULT_MPA * 1e6,
                P_exit=PE_DEFAULT_PA,
                alpha_min=0.40,
                alpha_max=0.90,
                coarse_points=7,
            )
            output = _format_result_text(result)
            self.root.after(0, self._on_success, output)
        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))

    def _on_success(self, output: str):
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, output)
        self.lbl_status.config(text="Готово.")
        self.btn_run.config(state=tk.NORMAL)

    def _on_error(self, err: str):
        self.lbl_status.config(text="Ошибка.")
        self.btn_run.config(state=tk.NORMAL)
        messagebox.showerror("Ошибка расчёта", err)


def run_gui():
    root = tk.Tk()
    RocketEquilibriumGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()

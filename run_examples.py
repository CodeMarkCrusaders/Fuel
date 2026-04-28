#!/usr/bin/env python3
# несколько тестовых расчётов для проверки решателя
# демонстрирует все три типа задач: TP, HP, SP — и запись лога итераций

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nasa9_parser import parse_thermo_file
from equilibrium import run_batch, print_result, find_thermo_db


def run_TP(name, reactants, T, P_atm, db, log_path=None):
    P = P_atm * 101325.0
    print(f"\n{'#'*70}")
    print(f"# [TP] {name}")
    print(f"# {reactants},  T={T:.0f} К,  P={P_atm} атм")
    print(f"{'#'*70}")

    t0 = time.time()
    result = run_batch(reactants, T=T, P=P, problem_type='TP',
                       verbose=False, log_path=log_path)
    print_result(result, db)
    print(f"  время: {time.time()-t0:.2f} с")
    if log_path:
        print(f"  лог: {log_path}")

    major = [(n, xi) for n, _, xi in result.get_gas_species() if xi > 1e-4]
    if major:
        print("  Основные компоненты:")
        for n, xi in major:
            print(f"    {n:<20} {xi*100:.4f} %")
    return result


def run_HP(name, reactants, H_target, P_atm, db, T_init=2500, log_path=None):
    P = P_atm * 101325.0
    print(f"\n{'#'*70}")
    print(f"# [HP] {name}")
    print(f"# {reactants},  H={H_target:.3e} Дж,  P={P_atm} атм,  T_init={T_init} К")
    print(f"{'#'*70}")

    t0 = time.time()
    result = run_batch(reactants, H=H_target, P=P, problem_type='HP',
                       T_init=T_init, verbose=False, log_path=log_path)
    print_result(result, db)
    print(f"  время: {time.time()-t0:.2f} с")
    if log_path:
        print(f"  лог: {log_path}")
    return result


def run_SP(name, reactants, S_target, P_atm, db, T_init=2500, log_path=None):
    P = P_atm * 101325.0
    print(f"\n{'#'*70}")
    print(f"# [SP] {name}")
    print(f"# {reactants},  S={S_target:.3e} Дж/К,  P={P_atm} атм,  T_init={T_init} К")
    print(f"{'#'*70}")

    t0 = time.time()
    result = run_batch(reactants, S=S_target, P=P, problem_type='SP',
                       T_init=T_init, verbose=False, log_path=log_path)
    print_result(result, db)
    print(f"  время: {time.time()-t0:.2f} с")
    if log_path:
        print(f"  лог: {log_path}")
    return result


def main():
    print("=" * 70)
    print("  Тестовые расчёты химического равновесия (TP / HP / SP)")
    print("=" * 70)

    db = parse_thermo_file(find_thermo_db())

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # ── TP-задачи (классические) ──────────────────────────────────────
    run_TP("H2 + O2 при 3000 К",       "2H2 + O2",          3000, 1.0, db,
           log_path=os.path.join(log_dir, 'tp_H2_O2_3000K.log'))
    run_TP("H2 + O2 при 1500 К",       "2H2 + O2",          1500, 1.0, db)
    run_TP("CH4 + воздух при 2000 К",  "1CH4 + 2O2 + 7.52N2", 2000, 1.0, db)
    run_TP("CO + O2 при 2500 К",       "2CO + O2",          2500, 1.0, db)

    # ── HP-задача: адиабатическое горение ────────────────────────────
    # для стехиометрической H2/O2: реагенты при 298 К имеют H ≈ 0,
    # поэтому ставим H_target = 0 — получим адиабатическую T пламени.
    run_HP("Адиабатическое пламя H2 + O2 (стехиометрия)",
           "2H2 + O2", H_target=0.0, P_atm=1.0, db=db, T_init=2500,
           log_path=os.path.join(log_dir, 'hp_H2_O2_adiabatic.log'))

    # стехиометрия CH4 + воздух: реагенты при 298 К  H_react ≈ Hf298(CH4) = -74900 Дж/моль
    run_HP("Адиабатическое пламя CH4 + воздух",
           "CH4 + 2O2 + 7.52N2", H_target=-74900.0, P_atm=1.0, db=db, T_init=2200,
           log_path=os.path.join(log_dir, 'hp_CH4_air_adiabatic.log'))

    # ── SP-задача: изэнтропическое расширение ────────────────────────
    # сначала найдём S при T=3000 К для 2H2+O2 при 1 атм, потом
    # пересчитаем равновесие при той же S и пониженном P (расширение в сопле).
    print("\n  --- готовим SP-задачу: считаем S при T=3000 К, P=10 атм ---")
    ref = run_batch("2H2 + O2", T=3000, P=10*101325.0, problem_type='TP', verbose=False)
    print(f"      S(3000 К, 10 атм) = {ref.entropy:.4e} Дж/К")
    run_SP("Изэнтропическое расширение H2/O2 до 1 атм",
           "2H2 + O2", S_target=ref.entropy, P_atm=1.0, db=db, T_init=2800,
           log_path=os.path.join(log_dir, 'sp_H2_O2_expansion.log'))

    print("\n" + "=" * 70)
    print("  Готово.")
    print(f"  Логи итераций: {log_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

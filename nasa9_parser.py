# парсер базы данных NASA-9 (формат CEA)
# читает thermo.inp и возвращает словарь веществ с их термодинамическими данными

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional  # noqa: F401  (Optional нужен для Species)


@dataclass
class TemperatureInterval:
    # один температурный интервал (обычно их 2-3 на вещество)
    T_low: float
    T_high: float
    n_coeff: int
    exponents: List[float]
    coeffs: List[float]      # a1..a7
    integration: List[float] # b1, b2 — константы интегрирования


@dataclass
class Species:
    # одно химическое вещество из базы данных
    name: str
    description: str
    n_intervals: int
    reference: str
    elements: Dict[str, float]  # {символ: кол-во атомов}
    phase: int       # 0=газ, 1=тв., 2=жидк.
    mol_weight: float
    hf298: float     # теплота образования при 298.15 К, Дж/моль
                     # для реагентов с n_intervals=0 — это assigned-h при T_assigned
    intervals: List[TemperatureInterval] = field(default_factory=list)
    is_reactant_only: bool = False
    # для реагентов с n_intervals=0 (например O2(L), H2(L)) — фиксированная T,
    # при которой задана энтальпия; полиномы по T отсутствуют, h(T)=hf298=const.
    T_assigned: Optional[float] = None

    @property
    def is_gas(self):
        return self.phase == 0

    @property
    def is_condensed(self):
        return self.phase != 0

    @property
    def is_tabular_only(self) -> bool:
        """Реагент задан только табличной энтальпией без полиномов."""
        return self.n_intervals == 0 and self.T_assigned is not None

    @property
    def phase_str(self):
        return {0: "газ", 1: "тв.", 2: "жидк."}.get(self.phase, "?")


def _parse_fortran_float(s: str) -> float:
    # в файле числа могут быть вида 1.23D+04 (фортран-стиль)
    s = s.strip()
    if not s:
        return 0.0
    s = s.replace('D', 'E').replace('d', 'e')
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_coefficient_line(line: str) -> List[float]:
    # каждый коэффициент занимает 16 символов, в строке их 5
    padded = line.ljust(80)
    return [_parse_fortran_float(padded[i*16:(i+1)*16]) for i in range(5)]


def parse_thermo_file(filepath: str) -> Dict[str, Species]:
    """Читает thermo.inp и возвращает словарь name -> Species."""
    species_db = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # ищем строку 'thermo' — данные начинаются через 2 строки после неё
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == 'thermo':
            data_start = i + 2
            break

    in_reactants_section = False
    idx = data_start

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if not stripped or stripped.startswith('!'):
            idx += 1
            continue

        if stripped.startswith('END PRODUCTS'):
            in_reactants_section = True
            idx += 1
            continue

        if stripped.startswith('END REACTANTS'):
            break

        # строка 1: имя + описание
        species_name = line[:18].strip()
        description = line[18:].strip()

        if not species_name:
            idx += 1
            continue

        idx += 1
        if idx >= len(lines):
            break

        # строка 2: состав, фаза, молярная масса, Hf298
        comp_line = lines[idx]
        if len(comp_line) < 50:
            idx += 1
            continue

        padded = comp_line.ljust(80)

        try:
            n_intervals = int(padded[0:2].strip())
        except (ValueError, IndexError):
            idx += 1
            continue

        reference = padded[3:9].strip()

        # 5 пар "символ + количество", каждая занимает 8 символов
        elements = {}
        for i in range(5):
            pos = 10 + i * 8
            symbol = padded[pos:pos+2].strip()
            try:
                count = float(padded[pos+2:pos+8].strip())
            except (ValueError, IndexError):
                count = 0.0
            if symbol and abs(count) > 1e-10 and symbol != 'E':
                elements[symbol.upper()] = count

        try:
            phase = int(padded[50:52].strip())
        except (ValueError, IndexError):
            phase = 0

        try:
            mol_weight = float(padded[52:65].strip())
        except (ValueError, IndexError):
            mol_weight = 0.0

        try:
            hf298 = float(padded[65:80].strip())
        except (ValueError, IndexError):
            hf298 = 0.0

        sp = Species(
            name=species_name,
            description=description,
            n_intervals=n_intervals,
            reference=reference,
            elements=elements,
            phase=phase,
            mol_weight=mol_weight,
            hf298=hf298,
            is_reactant_only=in_reactants_section,
        )

        idx += 1

        # n_intervals = 0 — реагент с табличной энтальпией (например O2(L)).
        # После comp-строки идёт одна строка вида "  90.170    0.0000   0.0 ..."
        # — T_assigned и нули (нет полиномов).  В таких случаях hf298 в comp-строке
        # — это и есть энтальпия при T_assigned (так делает CEA).
        if n_intervals == 0:
            if idx < len(lines):
                t_line = lines[idx].ljust(80)
                try:
                    T_assigned = float(t_line[0:11].strip())
                    if T_assigned > 0:
                        sp.T_assigned = T_assigned
                except (ValueError, IndexError):
                    pass
                idx += 1
            # такие записи сохраняем под именем и идём дальше
            if species_name not in species_db and sp.T_assigned is not None:
                species_db[species_name] = sp
            continue

        # читаем блоки температурных интервалов (по 3 строки каждый)
        for _ in range(n_intervals):
            if idx >= len(lines):
                break

            t_line = lines[idx].ljust(80)
            try:
                T_low = float(t_line[0:11].strip())
                T_high = float(t_line[11:22].strip())
                n_coeff = int(t_line[22:23].strip()) if t_line[22:23].strip() else 7
            except (ValueError, IndexError):
                idx += 1
                continue

            exponents = []
            for i in range(8):
                pos = 24 + i * 5
                try:
                    exponents.append(float(t_line[pos:pos+5].strip()))
                except (ValueError, IndexError):
                    exponents.append(0.0)

            idx += 1
            if idx >= len(lines):
                break

            first_coeffs = _parse_coefficient_line(lines[idx])
            idx += 1
            if idx >= len(lines):
                break

            second_coeffs = _parse_coefficient_line(lines[idx])
            idx += 1

            interval = TemperatureInterval(
                T_low=T_low,
                T_high=T_high,
                n_coeff=n_coeff,
                exponents=exponents,
                coeffs=first_coeffs[:5] + second_coeffs[:2],  # a1..a7
                integration=[second_coeffs[3], second_coeffs[4]],  # b1, b2
            )
            sp.intervals.append(interval)

        # дубликаты не добавляем
        if species_name not in species_db and n_intervals > 0:
            species_db[species_name] = sp

    return species_db


def get_products_for_elements(
    species_db: Dict[str, Species],
    element_set: set,
    include_condensed: bool = True,
    T: float = None,
) -> List[Species]:
    """Возвращает вещества из базы, подходящие как продукты реакции."""
    candidates = []

    for name, sp in species_db.items():
        if sp.is_reactant_only:
            continue
        # ионы пропускаем
        if '+' in name or '-' in name or name == 'e-':
            continue
        # все элементы вещества должны быть в нашем наборе
        if not set(sp.elements.keys()).issubset(element_set):
            continue
        if sp.is_condensed and not include_condensed:
            continue
        # проверяем что T попадает в диапазон данных (допуск ±100 К)
        if T is not None and sp.intervals:
            T_min = min(iv.T_low for iv in sp.intervals)
            T_max = max(iv.T_high for iv in sp.intervals)
            if T < T_min - 100 or T > T_max + 100:
                continue

        candidates.append(sp)

    return candidates


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python nasa9_parser.py <thermo.inp>")
        sys.exit(1)

    db = parse_thermo_file(sys.argv[1])
    print(f"загружено {len(db)} веществ")

    for name in ['H2O', 'O2', 'N2', 'CO2', 'CH4']:
        if name in db:
            sp = db[name]
            print(f"{sp.name}: M={sp.mol_weight:.3f}, Hf298={sp.hf298:.0f} Дж/моль, фаза={sp.phase_str}")

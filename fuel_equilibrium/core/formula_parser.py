# разбор химических формул и уравнений реакций
# поддерживает: H2O, Ca(OH)2, "2H2 + O2", "1.5H2 + 0.5N2" и т.д.

import re
from typing import Dict, List, Tuple


def parse_formula(formula: str) -> Dict[str, float]:
    """Разбирает формулу и возвращает {элемент: кол-во атомов}."""
    atom_counts = {}

    def _parse_group(s: str, mult: float = 1.0):
        i = 0
        while i < len(s):
            if s[i] == '(':
                # ищем закрывающую скобку
                depth, j = 1, i + 1
                while j < len(s) and depth > 0:
                    if s[j] == '(':   depth += 1
                    elif s[j] == ')': depth -= 1
                    j += 1
                inner = s[i+1:j-1]
                # число после скобки
                num = ''
                while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                    num += s[j]; j += 1
                _parse_group(inner, mult * (float(num) if num else 1.0))
                i = j
            elif s[i].isupper():
                # символ элемента
                sym = s[i]; i += 1
                while i < len(s) and s[i].islower():
                    sym += s[i]; i += 1
                # число после символа
                num = ''
                while i < len(s) and (s[i].isdigit() or s[i] == '.'):
                    num += s[i]; i += 1
                count = float(num) if num else 1.0
                canonical = sym[0].upper() + sym[1:].lower()
                atom_counts[canonical] = atom_counts.get(canonical, 0.0) + count * mult
            else:
                i += 1

    _parse_group(formula.strip())
    return atom_counts


def parse_reaction_string(reaction_str: str) -> List[Tuple[float, str, Dict[str, float]]]:
    """Разбирает левую часть уравнения реакции.

    Пример: "2H2 + O2" -> [(2.0, 'H2', {'H':2}), (1.0, 'O2', {'O':2})]
    """
    components = []
    for part in reaction_str.split('+'):
        part = part.strip()
        if not part:
            continue
        # пробуем отделить коэффициент от формулы
        m = re.match(r'^(\d+\.?\d*)\s*([A-Z].*)$', part)
        if m:
            coeff, formula = float(m.group(1)), m.group(2).strip()
        else:
            coeff, formula = 1.0, part
        components.append((coeff, formula, parse_formula(formula)))
    return components


def get_total_elements(components: List[Tuple[float, str, Dict[str, float]]]) -> Dict[str, float]:
    """Суммирует элементный состав всех реагентов."""
    total = {}
    for coeff, _, elems in components:
        for el, n in elems.items():
            total[el] = total.get(el, 0.0) + coeff * n
    return total


def format_elements(elements: Dict[str, float]) -> str:
    parts = []
    for el, n in sorted(elements.items()):
        r = round(n)
        if abs(n - r) < 1e-6:
            parts.append(f"{el}{r}" if r != 1 else el)
        else:
            parts.append(f"{el}{n:.4f}")
    return ' '.join(parts)


if __name__ == "__main__":
    tests = ["H2O", "CO2", "C2H5OH", "Ca(OH)2", "H2SO4"]
    print("Разбор формул:")
    for f in tests:
        print(f"  {f} -> {parse_formula(f)}")

    print("\nРазбор уравнений:")
    for r in ["2H2 + O2", "CH4 + 2O2", "C2H5OH + 3O2 + 11.28N2"]:
        total = get_total_elements(parse_reaction_string(r))
        print(f"  {r!r} -> {total}")

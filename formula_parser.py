"""
Разбор химических формул и уравнений реакций.

Умеет читать:
  - Простые формулы: H2O, CO2, CH4, NH3
  - Сложные формулы со скобками: Ca(OH)2, Al2(SO4)3
  - Уравнения левой части: "2H2 + O2", "CH4 + 2O2", "1.5 H2 + 0.5 N2"
"""

import re
from typing import Dict, List, Tuple


def parse_formula(formula: str) -> Dict[str, float]:
    """
    Разбирает химическую формулу и возвращает количество атомов каждого элемента.

    Примеры:
        'H2O'      -> {'H': 2.0, 'O': 1.0}
        'C2H5OH'   -> {'C': 2.0, 'H': 6.0, 'O': 1.0}
        'Ca(OH)2'  -> {'Ca': 1.0, 'O': 2.0, 'H': 2.0}
    """
    atom_counts: Dict[str, float] = {}

    def _parse_group(fragment: str, multiplier: float = 1.0):
        """Рекурсивно разбирает фрагмент формулы, учитывая скобки."""
        i = 0
        while i < len(fragment):
            char = fragment[i]

            if char == '(':
                # Ищем закрывающую скобку, учитывая вложенность
                depth = 1
                j = i + 1
                while j < len(fragment) and depth > 0:
                    if fragment[j] == '(':
                        depth += 1
                    elif fragment[j] == ')':
                        depth -= 1
                    j += 1
                # fragment[i+1 : j-1] — содержимое скобок
                inner = fragment[i + 1 : j - 1]

                # Читаем число после закрывающей скобки, например '2' в '(OH)2'
                num_str = ''
                while j < len(fragment) and (fragment[j].isdigit() or fragment[j] == '.'):
                    num_str += fragment[j]
                    j += 1
                group_multiplier = float(num_str) if num_str else 1.0

                _parse_group(inner, multiplier * group_multiplier)
                i = j

            elif char.isupper():
                # Начало символа элемента (заглавная буква)
                symbol = char
                i += 1
                # Строчные буквы — продолжение символа (например 'Ca', 'Fe')
                while i < len(fragment) and fragment[i].islower():
                    symbol += fragment[i]
                    i += 1

                # Число после символа элемента
                num_str = ''
                while i < len(fragment) and (fragment[i].isdigit() or fragment[i] == '.'):
                    num_str += fragment[i]
                    i += 1
                count = float(num_str) if num_str else 1.0

                # Приводим символ к каноническому виду: первая заглавная, остальные строчные
                canonical = symbol[0].upper() + symbol[1:].lower()
                atom_counts[canonical] = atom_counts.get(canonical, 0.0) + count * multiplier

            else:
                # Пропускаем посторонние символы (пробелы, запятые и т.п.)
                i += 1

    _parse_group(formula.strip())
    return atom_counts


def parse_reaction_string(
    reaction_str: str,
) -> List[Tuple[float, str, Dict[str, float]]]:
    """
    Разбирает левую часть уравнения реакции (список реагентов).

    Поддерживаемые форматы:
        "2H2 + O2"
        "CH4 + 2 O2"
        "1.5H2 + 0.5N2"
        "C2H5OH + 3O2 + 11.28N2"

    Возвращает список кортежей: (коэффициент, формула, {элемент: количество_атомов})
    """
    components = []

    # Разбиваем на отдельные слагаемые по символу '+'
    parts = reaction_str.split('+')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Пытаемся отделить числовой коэффициент от формулы
        # Принимаем форматы: "2H2O", "2 H2O", "2.5H2O", "H2O"
        match = re.match(r'^(\d+\.?\d*)\s*([A-Z].*)$', part)
        if match:
            coefficient = float(match.group(1))
            formula = match.group(2).strip()
        else:
            coefficient = 1.0
            formula = part

        element_dict = parse_formula(formula)
        components.append((coefficient, formula, element_dict))

    return components


def get_total_elements(
    components: List[Tuple[float, str, Dict[str, float]]],
) -> Dict[str, float]:
    """
    Считает суммарный элементный состав смеси реагентов.

    Аргументы:
        components: Результат parse_reaction_string()

    Возвращает словарь {символ_элемента: суммарное_количество_молей}
    """
    total: Dict[str, float] = {}
    for coefficient, formula, element_dict in components:
        for element, atom_count in element_dict.items():
            total[element] = total.get(element, 0.0) + coefficient * atom_count
    return total


def format_elements(elements: Dict[str, float]) -> str:
    """Форматирует словарь элементов в читаемую строку, например 'C2 H6 O'."""
    parts = []
    for element, count in sorted(elements.items()):
        rounded = round(count)
        if abs(count - rounded) < 1e-6:
            # Целое значение: 'H2', 'O' (единица не пишется)
            parts.append(f"{element}{rounded}" if rounded != 1 else element)
        else:
            # Дробное значение: 'H1.5000'
            parts.append(f"{element}{count:.4f}")
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Самотестирование при запуске напрямую
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Тест разбора формул ===")
    test_formulas = ["H2O", "CO2", "CH4", "C2H5OH", "N2O4", "NH3", "H2SO4", "Ca(OH)2"]
    for formula in test_formulas:
        print(f"  {formula:15s} -> {parse_formula(formula)}")

    print("\n=== Тест разбора уравнений ===")
    test_reactions = [
        "2H2 + O2",
        "CH4 + 2O2",
        "1.5H2 + 0.5N2",
        "C2H5OH + 3O2",
        "1CH4 + 2O2 + 7.52N2",
    ]
    for reaction in test_reactions:
        components = parse_reaction_string(reaction)
        total = get_total_elements(components)
        print(f"\n  '{reaction}':")
        for coeff, formula, elems in components:
            print(f"    {coeff} × {formula:12s} = {elems}")
        print(f"    Итого: {total}")

# -*- coding: utf-8 -*-
"""
Каталог топливных пар (как в RPA — Rocket Propulsion Analysis).

Содержит готовые «пресеты» окислителей и горючих, доступные пользователю
для быстрого выбора в интерактивном CLI или в CSV.  Все имена должны
точно совпадать с именами веществ в базе NASA-9 (data/thermo.inp).

Каждая запись содержит:
    name        — имя в базе NASA-9 (например, 'O2(L)')
    display     — отображаемое название ('Liquid Oxygen (LOX)')
    formula     — формула  ('O₂(L)')
    T_K         — рекомендуемая температура подачи (К). None означает
                  «использовать T_assigned из базы NASA-9» (для криогенов)
                  или 298.15 К для жидкостей при н.у.
    description — короткое описание
    aliases     — список псевдонимов, по которым пользователь может ввести
                  это вещество ('LOX', 'O2(L)', 'кислород жидкий', ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PropellantEntry:
    name: str                  # имя в базе NASA-9
    display: str               # отображаемое название
    formula: str               # химическая формула
    T_K: Optional[float] = None
    description: str = ""
    aliases: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Окислители
# ─────────────────────────────────────────────────────────────────────────────
OXIDIZERS: List[PropellantEntry] = [
    PropellantEntry(
        name="O2(L)", display="Liquid Oxygen (LOX)", formula="O₂(L)",
        T_K=None,  # T_assigned = 90.17 К
        description="Жидкий кислород, T_кип = 90.17 К",
        aliases=["LOX", "O2(L)", "O2L", "lox", "жидкий кислород", "кислород жидкий"],
    ),
    PropellantEntry(
        name="O2", display="Gaseous Oxygen (GOX)", formula="O₂",
        T_K=298.15,
        description="Газообразный кислород при н.у.",
        aliases=["GOX", "O2", "gox", "газообразный кислород"],
    ),
    PropellantEntry(
        name="N2O4(L)", display="Nitrogen Tetroxide (NTO)", formula="N₂O₄(L)",
        T_K=None,  # ≈ 298.15
        description="Тетраоксид азота — классический хранимый окислитель (АТ)",
        aliases=["NTO", "N2O4", "N2O4(L)", "АТ", "амил", "тетраоксид"],
    ),
    PropellantEntry(
        name="H2O2(L)", display="Hydrogen Peroxide (98%)", formula="H₂O₂(L)",
        T_K=None,  # T_assigned ≈ 298.15
        description="Перекись водорода (концентрированная) — пероксид",
        aliases=["H2O2", "H2O2(L)", "пероксид", "перекись водорода"],
    ),
    PropellantEntry(
        name="HNO3(L)", display="Nitric Acid (HNO₃)", formula="HNO₃(L)",
        T_K=None,
        description="Азотная кислота",
        aliases=["HNO3", "HNO3(L)", "азотная кислота", "АК"],
    ),
    PropellantEntry(
        name="IRFNA", display="IRFNA (Inhibited Red Fuming Nitric Acid)",
        formula="HNO₃ + N₂O₄ + HF",
        T_K=298.15,
        description="Ингибированная красная дымящая азотная кислота",
        aliases=["IRFNA", "АК-27И", "ингибированная азотная кислота"],
    ),
    PropellantEntry(
        name="F2(L)", display="Liquid Fluorine (LF₂)", formula="F₂(L)",
        T_K=None,  # ≈ 85 К
        description="Жидкий фтор — экспериментальный высокоэнергетический окислитель",
        aliases=["F2", "F2(L)", "LF2", "жидкий фтор", "фтор"],
    ),
    PropellantEntry(
        name="CLF3(L)", display="Chlorine Trifluoride (ClF₃)", formula="ClF₃(L)",
        T_K=None,
        description="Трифторид хлора — экзотический окислитель",
        aliases=["ClF3", "CLF3", "CLF3(L)", "трифторид хлора"],
    ),
    PropellantEntry(
        name="N2O(L),298.15K", display="Nitrous Oxide (N₂O)", formula="N₂O(L)",
        T_K=None,
        description="Закись азота",
        aliases=["N2O", "N2O(L)", "закись азота"],
    ),
    PropellantEntry(
        name="Air", display="Air", formula="Air (N₂+O₂+Ar+CO₂)",
        T_K=298.15,
        description="Воздух (для воздушно-реактивного режима)",
        aliases=["Air", "air", "воздух"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Горючие
# ─────────────────────────────────────────────────────────────────────────────
FUELS: List[PropellantEntry] = [
    PropellantEntry(
        name="H2(L)", display="Liquid Hydrogen (LH₂)", formula="H₂(L)",
        T_K=None,  # T_assigned = 20.27 К
        description="Жидкий водород, T_кип = 20.27 К",
        aliases=["LH2", "H2(L)", "H2L", "lh2", "жидкий водород", "водород жидкий"],
    ),
    PropellantEntry(
        name="H2", display="Gaseous Hydrogen (GH₂)", formula="H₂",
        T_K=298.15,
        description="Газообразный водород при н.у.",
        aliases=["GH2", "H2", "gh2", "газообразный водород"],
    ),
    PropellantEntry(
        name="CH4(L)", display="Liquid Methane (LCH₄)", formula="CH₄(L)",
        T_K=None,  # ≈ 111 К
        description="Жидкий метан, T_кип ≈ 111 К",
        aliases=["LCH4", "CH4(L)", "CH4L", "жидкий метан", "метан жидкий"],
    ),
    PropellantEntry(
        name="CH4", display="Gaseous Methane (GCH₄)", formula="CH₄",
        T_K=298.15,
        description="Газообразный метан",
        aliases=["CH4", "метан"],
    ),
    PropellantEntry(
        name="RP-1", display="RP-1 (Kerosene)", formula="≈ C₁H₁.₉₅₃",
        T_K=298.15,
        description="Керосин ракетный RP-1 (близкий аналог: T-1, РГ-1)",
        aliases=["RP-1", "RP1", "rp-1", "rp1", "kerosene", "керосин", "T-1", "РГ-1"],
    ),
    PropellantEntry(
        name="JP-4", display="JP-4 (Jet Fuel)", formula="≈ CH₂.₀₂",
        T_K=298.15,
        description="Авиационный керосин JP-4",
        aliases=["JP-4", "JP4", "jp-4", "jp4"],
    ),
    PropellantEntry(
        name="C2H8N2(L),UDMH", display="UDMH (НДМГ, Гептил)", formula="(CH₃)₂N₂H₂(L)",
        T_K=298.15,
        description="Несимметричный диметилгидразин (НДМГ, гептил)",
        aliases=["UDMH", "udmh", "НДМГ", "гептил", "Гептил", "C2H8N2", "C2H8N2(L),UDMH"],
    ),
    PropellantEntry(
        name="CH6N2(L)", display="MMH (Monomethylhydrazine)", formula="CH₃NHNH₂(L)",
        T_K=298.15,
        description="Монометилгидразин (ММГ)",
        aliases=["MMH", "mmh", "ММГ", "монометилгидразин", "CH6N2", "CH6N2(L)"],
    ),
    PropellantEntry(
        name="N2H4(L)", display="Hydrazine (N₂H₄)", formula="N₂H₄(L)",
        T_K=298.15,
        description="Гидразин",
        aliases=["N2H4", "N2H4(L)", "гидразин"],
    ),
    PropellantEntry(
        name="NH3(L)", display="Liquid Ammonia (NH₃)", formula="NH₃(L)",
        T_K=None,  # T_assigned ≈ 239.7 K
        description="Жидкий аммиак",
        aliases=["NH3", "NH3(L)", "жидкий аммиак", "аммиак"],
    ),
    PropellantEntry(
        name="CH3OH(L)", display="Methanol (CH₃OH)", formula="CH₃OH(L)",
        T_K=298.15,
        description="Метанол (метиловый спирт)",
        aliases=["MeOH", "CH3OH", "CH3OH(L)", "метанол", "метиловый спирт"],
    ),
    PropellantEntry(
        name="C2H5OH(L)", display="Ethanol (C₂H₅OH, 100%)", formula="C₂H₅OH(L)",
        T_K=298.15,
        description="Этанол (этиловый спирт) 100%",
        aliases=["EtOH", "C2H5OH", "C2H5OH(L)", "этанол", "этиловый спирт"],
    ),
    PropellantEntry(
        name="C3H8(L)", display="Liquid Propane (LPG)", formula="C₃H₈(L)",
        T_K=None,
        description="Жидкий пропан",
        aliases=["C3H8", "C3H8(L)", "пропан"],
    ),
    PropellantEntry(
        name="C2H6(L)", display="Liquid Ethane", formula="C₂H₆(L)",
        T_K=None,
        description="Жидкий этан",
        aliases=["C2H6", "C2H6(L)", "этан"],
    ),
    PropellantEntry(
        name="B5H9(L)", display="Pentaborane (Boroethane)", formula="B₅H₉(L)",
        T_K=298.15,
        description="Пентаборан — экзотическое высокоэнергетическое горючее",
        aliases=["B5H9", "B5H9(L)", "пентаборан"],
    ),
    PropellantEntry(
        name="B2H6(L)", display="Diborane (B₂H₆)", formula="B₂H₆(L)",
        T_K=None,
        description="Диборан",
        aliases=["B2H6", "B2H6(L)", "диборан"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# «Известные» популярные комбинации (для быстрого старта в CLI)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PropellantPair:
    oxidizer: str       # имя в базе
    fuel: str           # имя в базе
    name: str           # человекочитаемое название
    notes: str = ""     # пояснение

POPULAR_PAIRS: List[PropellantPair] = [
    PropellantPair("O2(L)", "H2(L)",
                   "LOX / LH₂",
                   "Кислород-водород: Шаттл (SSME), РД-0120, Vulcain, RS-25"),
    PropellantPair("O2(L)", "RP-1",
                   "LOX / RP-1 (керосин)",
                   "Кислород-керосин: F-1, РД-180/171, Merlin"),
    PropellantPair("O2(L)", "CH4(L)",
                   "LOX / LCH₄ (метан)",
                   "Кислород-метан: Raptor (SpaceX), BE-4 (Blue Origin)"),
    PropellantPair("O2(L)", "C2H5OH(L)",
                   "LOX / C₂H₅OH (этанол)",
                   "Кислород-этанол: V-2 (A-4), исторический"),
    PropellantPair("N2O4(L)", "C2H8N2(L),UDMH",
                   "АТ / НДМГ (Гептил)",
                   "Хранимое самовоспламеняющееся: РД-253 (Протон), РД-275"),
    PropellantPair("N2O4(L)", "CH6N2(L)",
                   "NTO / MMH",
                   "ВКС/ПКА: AJ10-190 (Shuttle OMS), MMH-NTO двигатели"),
    PropellantPair("N2O4(L)", "N2H4(L)",
                   "NTO / N₂H₄ (гидразин)",
                   "Хранимое: верхние ступени, апогейные двигатели"),
    PropellantPair("H2O2(L)", "RP-1",
                   "HTP / Керосин",
                   "Перекись / керосин: ТР-1 / British Black Knight"),
    PropellantPair("HNO3(L)", "C2H8N2(L),UDMH",
                   "АК / НДМГ",
                   "Азотная кислота / Гептил (старые ракеты)"),
    PropellantPair("IRFNA", "C2H8N2(L),UDMH",
                   "IRFNA / UDMH",
                   "Хранимое топливо ракет С-75/С-200/Scud"),
    PropellantPair("F2(L)", "H2(L)",
                   "F₂(L) / LH₂",
                   "Экспериментальное: высочайший Isp (~480с в вакууме)"),
    PropellantPair("CLF3(L)", "N2H4(L)",
                   "ClF₃ / N₂H₄",
                   "Эксзотика: очень агрессивный окислитель"),
    PropellantPair("N2O(L),298.15K", "C2H5OH(L)",
                   "N₂O / Этанол",
                   "Закись азота / этанол — любительские/гибридные двигатели"),
    PropellantPair("Air", "CH4(L)",
                   "Воздух / Метан",
                   "Воздушно-реактивный режим: ПВРД на метане"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Поиск/нормализация
# ─────────────────────────────────────────────────────────────────────────────

def _build_lookup(entries: List[PropellantEntry]) -> Dict[str, PropellantEntry]:
    """Строит словарь поиска по имени и всем псевдонимам (case-insensitive)."""
    lookup: Dict[str, PropellantEntry] = {}
    for e in entries:
        keys = [e.name, e.display] + list(e.aliases)
        for k in keys:
            kl = k.strip().lower()
            if kl and kl not in lookup:
                lookup[kl] = e
    return lookup


_OX_LOOKUP = _build_lookup(OXIDIZERS)
_FU_LOOKUP = _build_lookup(FUELS)


def resolve_propellant(
    user_input: str,
    kind: str = "any",
) -> Tuple[Optional[PropellantEntry], str]:
    """Пытается найти запись в каталоге по пользовательскому вводу.

    kind: 'oxidizer', 'fuel', 'any'.
    Возвращает (entry или None, имя в базе NASA-9 — пригодное для решателя).

    Если в каталоге не найдено — возвращает (None, user_input.strip()):
    мы доверяем пользователю — может, он указал имя из NASA-9 напрямую.
    """
    key = (user_input or "").strip().lower()
    if not key:
        return None, ""

    if kind in ("oxidizer", "any"):
        if key in _OX_LOOKUP:
            e = _OX_LOOKUP[key]
            return e, e.name
    if kind in ("fuel", "any"):
        if key in _FU_LOOKUP:
            e = _FU_LOOKUP[key]
            return e, e.name

    # ничего не нашли — возвращаем «как есть»
    return None, user_input.strip()


def list_oxidizers() -> List[PropellantEntry]:
    return list(OXIDIZERS)


def list_fuels() -> List[PropellantEntry]:
    return list(FUELS)


def list_popular_pairs() -> List[PropellantPair]:
    return list(POPULAR_PAIRS)


# ─────────────────────────────────────────────────────────────────────────────
# Печать каталога
# ─────────────────────────────────────────────────────────────────────────────

def print_catalog() -> None:
    """Печатает каталог в стиле РПА — для интерактивного выбора."""
    print()
    print("=" * 78)
    print("  Каталог топливных пар (RPA-style)")
    print("=" * 78)

    print("\n  ⬢ ПОПУЛЯРНЫЕ КОМБИНАЦИИ:")
    for i, p in enumerate(POPULAR_PAIRS, 1):
        print(f"   [{i:2d}]  {p.name:<32s}  — {p.notes}")

    print("\n  ⬢ ОКИСЛИТЕЛИ:")
    for i, e in enumerate(OXIDIZERS, 1):
        print(f"   [{i:2d}]  {e.display:<35s} ({e.formula:<16s}) "
              f"— {e.description}")

    print("\n  ⬢ ГОРЮЧИЕ:")
    for i, e in enumerate(FUELS, 1):
        print(f"   [{i:2d}]  {e.display:<35s} ({e.formula:<16s}) "
              f"— {e.description}")
    print()


if __name__ == "__main__":
    print_catalog()
    print("\n--- Проверка распознавания ---")
    for s in ["LOX", "lox", "жидкий кислород", "О2", "RP-1", "керосин", "Гептил",
              "MMH", "ММГ", "АТ", "NTO", "N2H4", "пероксид"]:
        e, nm = resolve_propellant(s)
        print(f"   {s!r:<25s} → {nm!r:<25s}  ({e.display if e else 'не найден в каталоге'})")

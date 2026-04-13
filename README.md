# Fuel Equilibrium

Расчёт равновесного состава продуктов горения методом минимизации энергии Гиббса.  
База данных — NASA CEA (9-коэффициентные полиномы), алгоритм Гордона–Макбрайда.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

**Интерактивный режим** — программа сама спросит реагенты, температуру и давление:

```bash
python equilibrium.py
```

**Из командной строки:**

```bash
python equilibrium.py -r "2H2 + O2" -T 3000 -P "1 atm"
python equilibrium.py -r "CH4 + 2O2 + 7.52N2" -T 2000 -P "1 bar" --verbose
```

**Из кода:**

```python
from equilibrium import run_batch

result = run_batch("2H2 + O2", T=3000, P=101325)
for name, moles, xi in result.get_gas_species():
    print(f"{name}: {xi:.2%}")
```

**Готовые примеры** (H2/O2, CH4/воздух, CO/O2 и др.):

```bash
python run_examples.py
```

## Структура проекта

```
├── equilibrium.py      # главный модуль, точка входа
├── gibbs_solver.py     # минимизация G методом SLSQP
├── nasa9_parser.py     # парсер базы данных thermo.inp
├── thermo_calc.py      # расчёт Cp, H, S, G по полиномам NASA-9
├── formula_parser.py   # разбор химических формул и уравнений
├── run_examples.py     # тестовые расчёты
└── data/
    ├── thermo.inp      # термодинамическая база данных NASA CEA
    ├── trans.inp       # транспортные свойства
    └── properties.inp  # свойства веществ
```

## Форматы ввода

Реагенты задаются как левая часть уравнения реакции:

```
2H2 + O2
CH4 + 2O2
C2H5OH + 3O2 + 11.28N2
1.5H2 + 0.5N2
```

Давление можно указать в разных единицах: `1 atm`, `1 bar`, `101325 Pa`, `100 kPa`.

## Флаги командной строки

| Флаг | Описание |
|------|----------|
| `-r` / `--reactants` | реагенты в кавычках |
| `-T` / `--temperature` | температура в кельвинах |
| `-P` / `--pressure` | давление с единицами |
| `--no-condensed` | не учитывать конденсированные фазы |
| `-v` / `--verbose` | подробный лог итераций |

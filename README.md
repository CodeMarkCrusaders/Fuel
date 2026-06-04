# Fuel Equilibrium + RPA-Style GUI

Расчёт равновесного состава продуктов горения методом минимизации энергии
Гиббса и графический интерфейс для расчёта газодинамики ракетного сопла в
стиле **RPA (Rocket Propulsion Analysis)** с тёмной темой.

> Этот репозиторий — результат рефакторинга: код больше не лежит «плоско»,
> а разбит на чёткие слои **core / rocket / io / gui**. Логика чистой физики
> отделена от I/O, прикладные ракетные расчёты — от ядра термодинамики, а
> GUI на PyQt5 — от расчётной части. См. [«Структура проекта»](#структура-проекта).

---

## Установка

```bash
pip install .            # установит пакет fuel_equilibrium со всеми зависимостями
# или, для разработки:
pip install -e .[gui,cea]
```

Опциональные дополнения:

* `[gui]` — PyQt5 + matplotlib для графического интерфейса;
* `[cea]` — Cantera для CEA-эквивалентного решателя.

---

## Структура проекта

```
fuel_equilibrium/
├── pyproject.toml              # метаданные пакета и зависимости
├── README.md                   # этот файл
│
└── fuel_equilibrium/
    ├── data/                   # термодинамические базы (NASA-9)
    │   ├── thermo.inp
    │   ├── trans.inp
    │   └── properties.inp
    │
    ├── core/                   # ЯДРО: чистая физика, без I/O и GUI
    │   ├── nasa9_parser.py     # парсер базы и dataclass Species
    │   ├── thermo_calc.py      # Cp, H, S, G по полиномам NASA-9
    │   ├── formula_parser.py   # разбор формул и уравнений реакций
    │   ├── gibbs_solver.py     # минимизация энергии Гиббса (TP / HP / SP)
    │   └── equilibrium.py      # высокоуровневые run_batch и find_thermo_db
    │
    ├── rocket/                 # ПРИКЛАДНОЙ СЛОЙ: расчёт ракетного сопла
    │   ├── nozzle_flow.py      # solve_rocket_nozzle и dataclass-ы Station/Performance
    │   └── cea_solver.py       # альтернатива на Cantera (опционально)
    │
    ├── io/                     # I/O и логирование
    │   ├── iteration_logger.py # IterationLogger / NullLogger
    │   ├── csv_runner.py       # batch-API через CSV (бывший rocket_csv.py)
    │   └── reporting.py        # print_result, print_nozzle_table
    │
    └── gui/                    # ИНТЕРФЕЙС: PyQt5
        ├── app.py              # главное окно (бывший rpa_gui.py)
        └── component_selector.py
```

---

## Поддерживаемые типы задач

| Тип  | Что задано | Что находится | Применение                            |
|------|------------|---------------|----------------------------------------|
| `TP` | T и P      | состав, H, S  | равновесие при заданных T, P           |
| `HP` | H и P      | T, состав, S  | адиабатическое горение                 |
| `SP` | S и P      | T, состав, H  | изэнтропическое расширение в сопле     |

---

## Запуск

### GUI (PyQt5, RPA-style)

```bash
python -m fuel_equilibrium.gui.app
```

Возможности:

* расчёт параметров (P, T, V, M, ρ, γ, состав) по длине сопла;
* **два решателя на выбор:** собственный (Gibbs + NASA-9) и CEA через Cantera;
* выбор компонентов топлива в стиле RPA — диалог с поиском, фильтрацией и
  смешиванием нескольких компонентов;
* тёмная тема Claude.ai;
* графики P-T, V-M, ρ-γₛ, профиль сопла, состав продуктов сгорания;
* экспорт результатов в CSV (Amesim-совместимый формат);
* сохранение/загрузка конфигурации (JSON);
* учёт криогенных топлив (`O2(L)`, `H2(L)`, `CH4(L)`) с автоматической
  `T_assigned`.

### CLI: расчёт равновесия

```bash
# Интерактивно (программа сама спросит тип задачи, реагенты и параметры):
python -m fuel_equilibrium.core.equilibrium

# TP — классическое равновесие
python -m fuel_equilibrium.core.equilibrium -r "2H2 + O2" -T 3000 -P "1 atm"

# HP — адиабатическое пламя (H_target в Дж)
python -m fuel_equilibrium.core.equilibrium -r "2H2 + O2" --HP -H 0 -P "1 atm" --T-init 2500

# SP — изэнтропическое расширение (S_target в Дж/К)
python -m fuel_equilibrium.core.equilibrium -r "2H2 + O2" --SP -S 562.6 -P "1 atm" --T-init 2500

# С записью журнала итераций
python -m fuel_equilibrium.core.equilibrium -r "CH4 + 2O2" -T 2000 -P "1 bar" --log run.log
```

### Из кода (Python API)

```python
from fuel_equilibrium.core import run_batch, find_thermo_db, parse_thermo_file
from fuel_equilibrium.io import print_result

# TP
res = run_batch("2H2 + O2", T=3000, P=101325, problem_type='TP')

# HP — адиабатическое горение
res = run_batch("2H2 + O2", H=0.0, P=101325, problem_type='HP', T_init=2500)
print(f"T_адиабатич = {res.T:.1f} К")

# SP — расширение в сопле, лог в файл
res = run_batch(
    "2H2 + O2", S=562.6, P=101325,
    problem_type='SP', T_init=2500,
    log_path="logs/expansion.log",
)

db = parse_thermo_file(find_thermo_db())
print_result(res, db)
```

### Расчёт ракетного сопла

```python
from fuel_equilibrium.core import parse_thermo_file, find_thermo_db
from fuel_equilibrium.rocket import Propellant, solve_rocket_nozzle
from fuel_equilibrium.io import print_nozzle_table

db = parse_thermo_file(find_thermo_db())

ox = Propellant("O2(L)", mass_kg=7.937)    # T_assigned = 90.17 К из базы
fu = Propellant("H2(L)", mass_kg=1.000)    # T_assigned = 20.27 К из базы

perf = solve_rocket_nozzle(
    oxidizer=ox, fuel=fu,
    P_chamber=10e6, P_exit=0.1013e6,
    species_db=db,
    n_intermediate_stations=3,
)
print_nozzle_table(perf)
print(f"Isp = {perf.Isp_s:.2f} с")
```

### Батч-расчёт сопла из CSV

```bash
python -m fuel_equilibrium.io.csv_runner examples/rocket_input_simple.csv out.csv
python -m fuel_equilibrium.io.csv_runner examples/rocket_input_ranges.csv out.csv --log-dir logs/csv
```

Входной CSV (semicolon-delimited, UTF-8 BOM) поддерживает как одиночные
значения, так и диапазоны `Pc_MPa_from / Pc_MPa_to / Pc_MPa_step`, которые
раскрываются через `itertools.product`.

---

## Алгоритм HP / SP

Двухуровневая схема:

1. **Внешний цикл** — поиск температуры методом секущей такой, чтобы
   `H(состав*, T) = H_target` (или `S(состав*, T, P) = S_target`).
2. **Внутренний цикл** — на каждом шаге внешнего цикла решается обычная
   TP-задача (минимизация G/RT методом SLSQP, резерв — trust-constr).

Используется **тёплый старт**: следующая внутренняя оптимизация начинается
из найденного на предыдущем шаге состава, что ускоряет сходимость в 3–5 раз.

---

## Расчёт сопла: что считается

Модуль `fuel_equilibrium.rocket.nozzle_flow` строит полную термодинамическую
и газодинамическую картину по сечениям сопла, аналогично RPA / NASA CEA:

| Сечение                            | Что находим                                       |
|------------------------------------|----------------------------------------------------|
| Injector (= Nozzle inlet, stagnation) | HP-задача: T, состав, S при заданной H реагентов |
| Nozzle throat (M = 1)              | поиск брентом по P; на каждом шаге — SP-задача    |
| Промежуточные сечения              | сетка по P между горловиной и срезом, SP-задача   |
| Nozzle exit                        | SP-задача при P = P_exit                          |

В каждом сечении считаются P, T, H, S, U; «замороженные» и «равновесные»
Cp, Cv, γ; изэнтропический показатель γₛ по NASA SP-273 (2.61); скорость
потока, скорость звука, M, ρ, MW, R_spec; `Ae/At`, mass flux; мольные и
массовые доли всех компонентов. Тяговые: `Isp`, `Isp_vac`, `C*`, `CF`.

### Эталонный тест H₂/O₂ (Pc = 10 МПа, Pe = 1 атм, α = 1)

| Параметр       | Наш собств. | Cantera (CEA) | RPA референс |
|----------------|------------:|--------------:|-------------:|
| T_chamber, К   | 3642.45     | 3652.08       | 3642.46      |
| Isp, с         | 374.34      | 374.94        | 374          |
| C*, м/с        | 2158.79     | 2162.25       | —            |
| V_exit, м/с    | 3670.97     | 3676.92       | 3670.95      |
| Aₑ/Aₜ          | 14.0145     | 14.0167       | 14.0118      |

---

## Форматы ввода

Реагенты задаются как левая часть уравнения реакции:

```
2H2 + O2
CH4 + 2O2
C2H5OH + 3O2 + 11.28N2
1.5H2 + 0.5N2
```

Давление: `1 atm`, `1 bar`, `101325 Pa`, `100 kPa`.

### Флаги CLI `equilibrium`

| Флаг | Описание |
|------|----------|
| `-r` / `--reactants`  | реагенты в кавычках |
| `-T` / `--temperature`| температура (К) — для TP |
| `-H` / `--enthalpy`   | энтальпия (Дж) — для HP |
| `-S` / `--entropy`    | энтропия (Дж/К) — для SP |
| `-P` / `--pressure`   | давление с единицами |
| `--HP`                | включает HP-задачу (требует `-H` и `-P`) |
| `--SP`                | включает SP-задачу (требует `-S` и `-P`) |
| `--T-init`            | начальная T для HP/SP (по умолч. 2000 К) |
| `--log`               | путь к файлу с журналом итераций |
| `--no-condensed`      | не учитывать конденсированные фазы |
| `-v` / `--verbose`    | подробный лог в консоль |

---

## Журнал итераций

Через флаг `--log <файл>` (CLI) или параметр `log_path=` (Python API) можно
сохранять подробный журнал хода решения: постановку задачи, список
кандидатов, `G/RT` и состав на каждой итерации SLSQP, шаги внешнего цикла
по T, невязки, итоговую сводку.

Пример строки журнала (HP-задача):

```
  >> внешний шаг   3:  T =  3070.9772 К,  H = -2.511224e+03  (target 0.000000e+00,  ΔT = -2.511e-03)
  [outer   3] iter     5:  G/RT = -7.90606275e+01,  невязка = 4.019e-14
        n[H2O                 ] = 1.720489e+00
        n[H2                  ] = 1.726538e-01
        ...
```

---

## Архитектурные принципы рефакторинга

* **`core/`** — никаких `print`, никаких файловых артефактов, никакого
  GUI. Только чистая термодинамика + математика.
* **`rocket/`** — прикладной слой поверх `core/`. Структуры данных
  (Propellant / StationResult / RocketPerformance) и алгоритм
  газодинамики сопла. Зависит от `core` и `io.iteration_logger`.
* **`io/`** — всё, что про побочные эффекты: логи итераций,
  CSV-batch-API, форматирование таблиц. Зависит от `core` и `rocket`.
* **`gui/`** — единственный потребитель PyQt5 и matplotlib. Зависит
  от всех нижних слоёв, но никто не зависит от него.

Циклических зависимостей нет: `core → io ← rocket ← gui` (стрелки = «зависит от»).

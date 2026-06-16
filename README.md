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

### Развёртка по соотношению компонентов O/F (поиск оптимума Isp)

Модуль `fuel_equilibrium.rocket.of_sweep` строит классическую кривую
**«Isp vs O/F»** (как в RPA / NASA CEA): для серии массовых соотношений
окислитель/горючее запускается равновесный расчёт сопла, собираются кривые
`Isp`, `Isp_vac`, `C*`, `T_chamber`, `CF`, после чего находится O/F,
максимизирующий удельный импульс. Оптимум уточняется параболической
интерполяцией по трём узлам вокруг максимума сетки, поэтому он не «прилипает»
к узлам, а попадает между ними.

```python
from fuel_equilibrium.core import parse_thermo_file, find_thermo_db
from fuel_equilibrium.rocket import sweep_of_ratio
from fuel_equilibrium.io import print_of_sweep_table

db = parse_thermo_file(find_thermo_db())

sweep = sweep_of_ratio(
    oxidizer_name="O2(L)", fuel_name="H2(L)",
    P_chamber=10e6, P_exit=0.1013e6,
    species_db=db,
    of_min=3.0, of_max=8.0, n_points=6,   # либо of_values=[4.0, 5.0, 6.0]
    optimize_for="Isp",                    # или "Isp_vac"
)

print_of_sweep_table(sweep)
print(f"Оптимум по Isp:     O/F = {sweep.best_of:.3f},  Isp = {sweep.best_Isp_s:.2f} с")
print(f"Оптимум по Isp_vac: O/F = {sweep.best_of_vac:.3f},  Isp_vac = {sweep.best_Isp_vac_s:.2f} с")
```

Для H₂/O₂ при `Pc = 10 МПа`, `Pe = 1 атм` это даёт оптимум по импульсу на
срезе при **O/F ≈ 4.3** и по вакуумному импульсу при **O/F ≈ 4.6** — в
согласии со справочными данными по кислородно-водородным ЖРД.

Особенности:

* сетка задаётся либо равномерно (`of_min` / `of_max` / `n_points`), либо
  явным списком значений (`of_values`, имеет приоритет);
* `fuel_mass_kg` задаёт базовую массу горючего, масса окислителя берётся как
  `O/F · fuel_mass_kg`;
* `oxidizer_T_K` / `fuel_T_K` — температуры подачи (по умолчанию из базы, в
  т. ч. криогенные `T_assigned`);
* точки, где расчёт упал, помечаются полем `error` и не участвуют в поиске
  оптимума (вся развёртка не падает из-за одной точки);
* в результате `OFSweepResult` каждая точка (`OFSweepPoint`) хранит полный
  объект `RocketPerformance` (`performance`) — можно построить любой график
  или вытащить состав/сечения для выбранного O/F.

### Построение геометрии сопла (учебник Добровольского, гл. 2)

Модуль `fuel_equilibrium.rocket.nozzle_geometry` строит контур сопла по
методике учебника **М. В. Добровольский, «Жидкостные ракетные двигатели.
Основы проектирования» (2016), глава 2 «Сопла ЖРД»**. Поддерживаются два
типа сопел с настройкой **всех** параметров:

* **Коническое сопло (§2.3)** — дозвуковой конус θ_вх (2θ_вх = 45…80°),
  скругление R_скр перед горловиной, скругление R_1 на входе из камеры,
  скругление r_скр за горловиной и прямой сверхзвуковой конус θ_a
  (2θ_a = 25…30°).
* **Профилированное (укороченное оптимальное) сопло (§2.6)** — та же
  дозвуковая часть, а сверхзвуковая строится параболой A_n → C
  (квадратичная кривая Безье через точку пересечения касательных f) с углом
  θ_m в начале и θ_a на срезе. Углы и длина берутся из семейства оптимальных
  контуров (**Рис. 2.14**, γ = 1.23) по степени расширения, либо задаются
  вручную.

Дополнительно реализованы формулы §2.2–2.3:

* φ_рас = (1 + cos θ_a)/2 — коэффициент рассеяния потока на срезе;
* θ_a из условия безотрывного течения при недорасширении
  (ур. 2.23/2.24): `sin 2θ_a = (p_a − p_н)/(½ρ_a w_a²)·ctg μ_a`, `sin μ_a = 1/M_a`.

```python
from fuel_equilibrium.rocket import (
    build_conical_nozzle, build_profiled_nozzle,
    build_geometry_from_performance, dispersion_loss_coeff,
)

# Профилированное сопло: R_кр=50 мм, F_a/F_кр=16 (углы из Рис.2.14)
g = build_profiled_nozzle(0.05, 16.0)
print(f"θ_m={g.theta_max_deg:.1f}°  θ_a={g.theta_exit_deg:.1f}°  "
      f"L={g.length_total_m*1e3:.1f} мм  φ_рас={g.phi_dispersion:.4f}")

# Коническое сопло с явными параметрами
c = build_conical_nozzle(
    0.05, 16.0, theta_exit_deg=15.0, theta_in_deg=30.0,
    R_round_sub_factor=1.0, R1_inlet_factor=3.0, r_round_sup_factor=0.45,
)

# Контур (x, r) для отрисовки/экспорта
x, r = g.as_xy_arrays()

# Построить геометрию прямо из результата solve_rocket_nozzle:
geom = build_geometry_from_performance(
    perf, R_throat_m=0.05, method="profiled", p_ambient_Pa=101325,
)
```

В **GUI** настройка геометрии доступна в двух местах:

* **При расчёте сопла** — в левой панели ввода появилась группа
  **«Профиль сопла (Добровольский, гл. 2)»**: можно **выбрать тип сопла**
  (коническое / профилированное) и **вручную задать геометрические
  параметры** (R_кр с единицами, R_камеры/R_кр, углы θ_вх / θ_m / θ_a,
  множители скруглений R_скр / r_скр / R_1, режим «авто» из Рис. 2.14 для
  профилированного). После нажатия «Рассчитать сопло» график
  **«Профиль сопла»** строится именно по выбранному типу и параметрам, а
  степень расширения F_a/F_кр берётся из результата газодинамического
  расчёта. Все эти настройки сохраняются/загружаются в JSON-конфигурации.
* **Отдельная вкладка «Геометрия сопла (Добровольский)»** — то же самое плюс
  интерактивная отрисовка, текстовая сводка, экспорт точек контура в CSV и
  кнопка «Взять F_a/F_кр из расчёта».

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

# Fuel Equilibrium

Расчёт равновесного состава продуктов горения методом минимизации энергии Гиббса.
Полный термодинамический и газодинамический расчёт ракетного сопла —
аналог RPA / NASA CEA. Включает:

- авто-определение оптимального соотношения компонентов топлива (по Isp / Isp_vac / C* / Tc);
- задание соотношения через **O/F**, **α** (коэф. избытка окислителя) или **оптимум**;
- задание давления в любых единицах: **МПа, Па, атм, кгс/см², бар, фунт/дюйм² (psi)**, kPa, mmHg;
- **каталог топливных пар** (LOX/LH₂, LOX/RP-1, NTO/UDMH, …) с RPA-подобным выбором.

Поддерживаемые типы задач:

| Тип  | Что задано     | Что находится         | Применение                         |
|------|----------------|-----------------------|-------------------------------------|
| `TP` | T и P          | состав, H, S          | равновесие при заданных T, P       |
| `HP` | H и P          | T, состав, S          | адиабатическое горение             |
| `SP` | S и P          | T, состав, H          | изэнтропическое расширение в сопле |

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

**Интерактивный режим** — программа сама спросит тип задачи, реагенты, параметры и путь к логу:

```bash
python equilibrium.py
```

**Из командной строки:**

```bash
# TP-задача: классическое равновесие
python equilibrium.py -r "2H2 + O2" -T 3000 -P "1 atm"

# HP-задача: адиабатическое пламя (H_target в Дж)
python equilibrium.py -r "2H2 + O2" --HP -H 0 -P "1 atm" --T-init 2500

# SP-задача: изэнтропическое расширение (S_target в Дж/К)
python equilibrium.py -r "2H2 + O2" --SP -S 562.6 -P "1 atm" --T-init 2500

# с записью всех итераций в журнал
python equilibrium.py -r "CH4 + 2O2" -T 2000 -P "1 bar" --log run.log
```

**Из кода:**

```python
from equilibrium import run_batch

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

# у результата есть и состав, и термодинамические свойства смеси:
for name, moles, xi in res.get_gas_species():
    print(f"{name}: {xi:.2%}")
print(f"H = {res.enthalpy:.4e} Дж,  S = {res.entropy:.4e} Дж/К")
```

**Готовые примеры** (TP / HP / SP, H2/O2, CH4/воздух, CO/O2):

```bash
python run_examples.py
```

Логи всех итераций сохраняются в папке `logs/`.

## Журнал итераций (логирование)

Через флаг `--log <файл>` (CLI) или параметр `log_path=` (Python API) можно
сохранять подробный журнал всего хода решения. В лог попадают:

- постановка задачи (тип, реагенты, T/P/H/S, элементный баланс);
- список веществ-кандидатов;
- значения целевой функции `G/RT` и текущий состав на **каждой**
  внутренней итерации SLSQP;
- все шаги внешнего цикла по T (для HP/SP);
- невязки балансов;
- итоговый состав смеси и сводка.

Пример строки журнала (HP-задача):

```
  >> внешний шаг   3:  T =  3070.9772 К,  H = -2.511224e+03  (target 0.000000e+00,  ΔT = -2.511e-03)
  [outer   3] iter     5:  G/RT = -7.90606275e+01,  невязка = 4.019e-14
        n[H2O                 ] = 1.720489e+00
        n[H2                  ] = 1.726538e-01
        ...
```

## Структура проекта

```
├── equilibrium.py            # главный модуль равновесия (CLI + интерактив для TP/HP/SP)
├── gibbs_solver.py           # минимизация G; решатели TP / HP / SP
├── iteration_logger.py       # логгер итераций (запись в файл)
├── nasa9_parser.py           # парсер базы данных thermo.inp (+ T_assigned для криогенов)
├── thermo_calc.py            # расчёт Cp, H, S, G по полиномам NASA-9
├── formula_parser.py         # разбор химических формул и уравнений
├── nozzle_flow.py            # ракетное сопло: HP в камере, SP по сечениям, M=1
├── units.py                  # парсер давления: МПа, Па, атм, кгс/см², бар, psi, …
├── propellants_catalog.py    # КАТАЛОГ ТОПЛИВНЫХ ПАР (RPA-style) с псевдонимами
├── propellant_optimizer.py   # АВТО-ПОИСК оптимального O/F (по Isp / Isp_vac / C* / Tc)
├── rpa_cli.py                # ИНТЕРАКТИВНЫЙ CLI в стиле РПА (выбор пары + режим)
├── rocket_csv.py             # батч-CSV драйвер (с поддержкой OF / alpha / optimal)
├── run_examples.py           # тестовые расчёты (все три типа задач)
├── examples/                 # примеры входных CSV для rocket_csv.py
│   ├── rocket_input_simple.csv      # одиночный расчёт (legacy формат)
│   ├── rocket_input_ranges.csv      # диапазоны по Pc/Pe/α  (legacy формат)
│   ├── rocket_input_units.csv      # разные единицы давления
│   └── rocket_input_optimal.csv    # АВТО-ПОИСК оптимального O/F
└── data/
    ├── thermo.inp            # термодинамическая база данных NASA CEA
    ├── trans.inp             # транспортные свойства
    └── properties.inp        # свойства веществ
```

## RPA-подобный интерфейс (новое)

### Интерактивный CLI с выбором топливной пары

```bash
python rpa_cli.py
```

Программа спросит:

1. **Топливная пара**  — выбор из готовых пресетов (LOX/LH₂, LOX/RP-1,
   NTO/НДМГ, NTO/MMH, F₂/LH₂, и т.д.) **либо** ввод окислителя и горючего
   по аббревиатуре (`LOX`, `LH2`, `НДМГ`, `MMH`, `керосин`, `Гептил`, …).
2. **Давление в камере и на срезе** — можно указать в любых единицах:
   `10 MPa`, `100 atm`, `1500 psi`, `100 kgf/cm²`, `100 bar`, `0.1013 МПа`, …
3. **Способ задания соотношения**:
   - **O/F**  — массовое соотношение (m_окислителя / m_горючего);
   - **α**    — коэффициент избытка окислителя = (O/F) / (O/F)_стехиом.;
   - **оптимальное** — авто-поиск максимума **Isp**, **Isp_vac**, **C\*** или **T_camera**.

Просмотр каталога топливных пар:

```bash
python rpa_cli.py --catalog
```

### Авто-определение оптимального O/F программно

```python
from nasa9_parser import parse_thermo_file
from equilibrium import find_thermo_db
from propellant_optimizer import (
    RatioSpec, find_optimal_OF, print_optimization_summary,
)

db = parse_thermo_file(find_thermo_db())

# Авто-оптимум по Isp на срезе
spec = RatioSpec(
    mode="optimal", target="Isp",        # Isp / Isp_vac / Cstar / T_chamber
    alpha_min=0.4, alpha_max=1.2, n_grid=9,
    refine=True,                          # уточнение параболической интерполяцией
)
res = find_optimal_OF(
    oxidizer_name="O2(L)", fuel_name="H2(L)",
    spec=spec,
    P_chamber_Pa=10e6, P_exit_Pa=0.1013e6,
    species_db=db,
)
print_optimization_summary(res)
print(f"Оптимум:  α = {res.alpha:.4f},  O/F = {res.OF:.4f},  Isp = {res.target_value:.2f} с")
```

Результат — кривая Isp(α) и точка максимума. Для H₂/O₂ при Pc=10 МПа,
Pe=1 атм получается α_opt ≈ 0.535, O/F_opt ≈ 4.246, Isp ≈ 399.8 с
(полностью соответствует RPA / NASA CEA).

### Задание соотношения через O/F или α

```python
# Через O/F
spec = RatioSpec(mode="OF", value=4.5)

# Через коэффициент избытка окислителя α
spec = RatioSpec(mode="alpha", value=1.0)   # 1.0 = стехиометрия
```

### Единицы давления

Поддерживается **парсер единиц давления** (модуль `units.py`):

```python
from units import parse_pressure

parse_pressure("10 MPa")        # → 10 000 000 Pa
parse_pressure("100 atm")       # → 10 132 500 Pa
parse_pressure("100 kgf/cm²")   # → 9 806 650 Pa
parse_pressure("1500 psi")      # → 10 342 135.94 Pa
parse_pressure("100 bar")       # → 10 000 000 Pa
parse_pressure("10 МПа")        # русские эквиваленты тоже работают
parse_pressure("100 кгс/см²")
parse_pressure("14.7 фунт/дюйм²")
```

Все эти строки можно подставить в CLI и в CSV (см. ниже).

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
| `-T` / `--temperature` | температура (К) — для TP |
| `-H` / `--enthalpy` | энтальпия (Дж) — для HP |
| `-S` / `--entropy` | энтропия (Дж/К) — для SP |
| `-P` / `--pressure` | давление с единицами |
| `--HP` | включает HP-задачу (требует `-H` и `-P`) |
| `--SP` | включает SP-задачу (требует `-S` и `-P`) |
| `--T-init` | начальная T для внешних итераций HP/SP (по умолч. 2000 К) |
| `--log` | путь к файлу с журналом итераций |
| `--no-condensed` | не учитывать конденсированные фазы |
| `-v` / `--verbose` | подробный лог в консоль |

## Алгоритм HP / SP

Для HP- и SP-задач реализован **двухуровневый алгоритм**:

1. **Внешний цикл** — поиск температуры методом секущей такой, чтобы
   `H(состав*, T) = H_target`  (или `S(состав*, T, P) = S_target`).
2. **Внутренний цикл** — на каждом шаге внешнего цикла решается обычная
   TP-задача (минимизация G/RT методом SLSQP с резервом trust-constr).

Используется **тёплый старт**: следующая внутренняя оптимизация начинается
из найденного на предыдущем шаге состава, что ускоряет сходимость в 3–5 раз.

## Расчёт ракетного сопла (равновесное течение)

Модуль `nozzle_flow.py` строит полную термодинамическую и газодинамическую
картину по сечениям сопла — аналогично RPA / NASA CEA:

| Сечение | Что находим |
|---------|-------------|
| Injector (= Nozzle inlet, stagnation) | HP-задача: T, состав, S при заданной H реагентов |
| Nozzle throat (M = 1) | поиск брентом по P; на каждом шаге — SP-задача |
| Промежуточные сечения | сетка по P между горловиной и срезом, SP-задача |
| Nozzle exit | SP-задача при P = P_exit |

В каждом сечении считаются:

- P, T, H, S, U (на 1 кг смеси);
- **«замороженные»** и **«равновесные»** Cp, Cv, γ;
- **изэнтропический показатель** γₛ по NASA SP-273 (2.61):
  `γₛ = −1 / [(∂lnV/∂lnP)_T + nR(∂lnV/∂lnT)²_P / Cp_eq]`
- скорость потока V из энерговского баланса `V = √(2·(H_camera − H))`;
- скорость звука `a = √(γₛ·R_spec·T)`, число Маха `M = V/a`;
- плотность ρ, молярная масса MW, газовая постоянная R_spec;
- площадь сечения `Ae/At = (ρ_t·V_t)/(ρ·V)`, mass flux ρV;
- мольные и массовые доли всех компонентов в равновесии.

Тяговые характеристики: `Isp`, `Isp_vac`, `C*`, `CF`.

### Программно

```python
from nasa9_parser import parse_thermo_file
from equilibrium import find_thermo_db
from nozzle_flow import Propellant, solve_rocket_nozzle, print_nozzle_table

db = parse_thermo_file(find_thermo_db())

# криогенные O2(L)@90.17 К и H2(L)@20.27 К берут T_assigned из базы
ox = Propellant("O2(L)", mass_kg=7.937)
fu = Propellant("H2(L)", mass_kg=1.000)

perf = solve_rocket_nozzle(
    oxidizer=ox, fuel=fu,
    P_chamber=10e6, P_exit=0.1013e6,
    species_db=db,
    n_intermediate_stations=3,  # ещё 3 точки между горловиной и срезом
)
print_nozzle_table(perf)        # таблица в стиле RPA / CEA
print(f"Isp = {perf.Isp_s:.2f} с")
```

Тест на H₂+O₂ при Pc = 10 МПа, α = 1 (α = O/F / O/F_st = 1) воспроизводит
референсные данные RPA до 4–5 знаков:

| Параметр | Наш | RPA |
|----------|----:|----:|
| T_chamber, К | 3642.45 | 3642.46 |
| Cp_eq (eq.), кДж/(кг·К) | 9.4244 | 9.4244 |
| Cv_eq (eq.), кДж/(кг·К) | 8.0044 | 8.0043 |
| γ_eq | 1.1774 | 1.1774 |
| γₛ (Isentropic exp.) | 1.1315 | 1.1315 |
| V_exit, м/с | 3670.97 | 3670.95 |
| M_exit | 3.2553 | 3.2552 |
| Aₑ/Aₜ | 14.0145 | 14.0118 |
| Isp, с | 374.34 | — |
| C*, м/с | 2158.79 | — |

## Батч-расчёт ракетного сопла из CSV

`rocket_csv.py` — обёртка над `solve_rocket_nozzle` / `find_optimal_OF`,
читающая входной CSV с диапазонами параметров `Pc`, `Pe`, `O/F` или `α`
(а также с **авто-поиском оптимального соотношения**) и пишущая
полный отчёт по сечениям сопла в выходной CSV.

```bash
python rocket_csv.py examples/rocket_input_simple.csv   out.csv
python rocket_csv.py examples/rocket_input_ranges.csv   out.csv --log-dir logs/csv
python rocket_csv.py examples/rocket_input_units.csv    out.csv
python rocket_csv.py examples/rocket_input_optimal.csv  out.csv
```

### 1) Простой вариант (legacy) — давление в МПа, заданная α

```csv
oxidizer;fuel;oxidizer_temp_K;fuel_temp_K;Pc_MPa;Pe_MPa;alpha;n_intermediate_stations;top_species
O2(L);H2(L);;;10;0.101325;opt;0;12
```

### 2) Любые единицы давления и аббревиатуры топлив (RPA-style)

```csv
oxidizer;fuel;Pc;Pe;ratio_mode;alpha;n_intermediate_stations;top_species
LOX;LH2;10 MPa;1 atm;alpha;1.0;0;10
LOX;LH2;100 bar;100 kPa;alpha;1.0;0;10
LOX;LH2;100 kgf/cm2;14.7 psi;alpha;1.0;0;10
LOX;LH2;1450 psi;1 atm;alpha;1.0;0;10
```

В колонках `oxidizer` / `fuel` можно использовать **аббревиатуры из каталога**:
`LOX`, `LH2`, `LCH4`, `RP-1`, `NTO`, `НДМГ` / `UDMH` / `Гептил`, `MMH`, `ММГ`,
`N2H4`, `H2O2`, `Air`, …. Они автоматически переводятся в имена базы NASA-9.

### 3) Авто-поиск оптимального соотношения компонентов

```csv
oxidizer;fuel;Pc;Pe;ratio_mode;optimize_target;alpha_min;alpha_max;n_grid;n_intermediate_stations;top_species
LOX;LH2;10 MPa;1 atm;optimal;Isp;0.4;1.2;9;0;10
```

При `ratio_mode=optimal` решатель **сам найдёт максимум** по заданной
целевой функции (`Isp`, `Isp_vac`, `Cstar`, `T_chamber`).  В выходной CSV
для каждого кейса попадают:

- основные сечения сопла в оптимальной точке (Injector / inlet / throat /
  exit / промежуточные);
- **полная таблица скана** по α (по одной строке на точку сетки) с меткой
  `OPT_SCAN  α=<value>` — удобно для построения кривых Isp(α).

### 4) Задание O/F напрямую

```csv
oxidizer;fuel;Pc;Pe;ratio_mode;OF;n_intermediate_stations;top_species
LOX;RP-1;10 MPa;1 atm;OF;2.27;0;10
N2O4(L);UDMH;10 MPa;1 atm;OF;2.67;0;10
```

### 5) Диапазоны через from/to/step (с любыми единицами)

```csv
oxidizer;fuel;Pc_from;Pc_to;Pc_step;Pe;ratio_mode;alpha_from;alpha_to;alpha_step;n_intermediate_stations;top_species
LOX;LH2;8 MPa;12 MPa;2 MPa;1 atm;alpha;0.8;1.2;0.2;2;10
```

(Эта строка раскрывается в 3 × 1 × 3 = **9 кейсов**.)

### Доступные ключи CSV

| Колонка                        | Описание |
|---|---|
| `oxidizer`, `fuel`             | имя из NASA-9 или аббревиатура (`LOX`, `НДМГ`, …) |
| `oxidizer_temp_K`, `fuel_temp_K` | температура подачи, К (пусто → из базы / 298 К) |
| `Pc`, `Pe`                     | давление со строкой единиц, напр. `10 MPa`, `1 atm`, `1500 psi` |
| `Pc_unit`, `Pe_unit`           | (опц.) единицы, если давление задано числом |
| `Pc_from/_to/_step`, `Pe_from/_to/_step` | диапазон давлений (поддерживает единицы) |
| `Pc_MPa`, `Pe_MPa` (+ `_from/_to/_step`) | legacy: давление в МПа |
| `ratio_mode`                   | `OF` / `alpha` / `optimal` (по умолч. `alpha`) |
| `alpha` (+ `_from/_to/_step`)  | коэффициент избытка окислителя |
| `OF` (+ `_from/_to/_step`)     | массовое соотношение окислитель/горючее |
| `optimize_target`              | `Isp` / `Isp_vac` / `Cstar` / `T_chamber` |
| `alpha_min`, `alpha_max`, `n_grid`, `refine` | параметры скана для `optimal` |
| `n_intermediate_stations`      | число промежуточных сечений между throat и exit |
| `top_species`                  | сколько самых распространённых компонентов выводить |

Если `oxidizer_temp_K` / `fuel_temp_K` пустые, температура берётся из
NASA-базы (`T_assigned` для криогенных реагентов вроде `O2(L)`, `H2(L)`)
или 298.15 К по умолчанию.

Для `rocket_csv.py` включён быстрый режим ракетного расчёта:
- фиксируются 4 сечения: `Injector`, `Nozzle inlet`, `Nozzle throat`, `Nozzle exit`;
- если `alpha` не задана (или `alpha=opt`/`auto`) — выполняется ускорённый
  поиск оптимального массового соотношения O/F по максимуму `Isp`.
- список выбираемых компонентов топлива очищен от ионов.

**Выходной CSV** — одна строка на каждое сечение каждого кейса (плюс одна
строка с описанием ошибки, если кейс не сошёлся).  Колонки:

- идентификатор кейса (`Глобальный_ID_варианта`, `Строка_входного_CSV`, `Номер_варианта`);
- параметры топлива (`Окислитель`, `Горючее`, температуры, Pc, Pe, α, φ, O/F);
- тяговые характеристики (`Isp`, `Isp_vac`, `C*`, `CF`);
- параметры сечения (`Сечение`, `Метка`, `Давление_МПа`, `Температура_К`,
  `Энтальпия_кДж_кг`, `Энтропия_кДж_кгК`, `Cp_eq`, `Cv_eq`, `Gamma_eq`,
  `Изэнтропический_показатель`, MW, R_spec, ρ, скорость звука, V, M,
  `Ae/At`, mass flux);
- мольные `x_*` и массовые `w_*` доли топ-N компонентов (общий список для
  всех кейсов файла, чтобы все строки имели одинаковые колонки).

С флагом `--log-dir <папка>` для каждого кейса будет создан подробный
журнал решателя `case_NNNN.log`.

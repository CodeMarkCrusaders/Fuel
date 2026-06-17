"""Тесты подавления численного шума в профилях параметров по оси сопла."""

import numpy as np
import pytest

from fuel_equilibrium.rocket.signal_cleanup import (
    pava_increasing, monotone_despike, hampel_filter,
)


def _peakiness(y):
    """Мера «пилообразности»: макс. модуль 2-й разности, нормированный на размах."""
    y = np.asarray(y, float)
    rng = float(np.nanmax(y) - np.nanmin(y)) + 1e-12
    return float(np.max(np.abs(np.diff(y, 2)))) / rng


# ─────────────────────────────────────────────────────────────────────────────
# PAVA — изотоническая регрессия
# ─────────────────────────────────────────────────────────────────────────────

def test_pava_already_increasing_unchanged():
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = pava_increasing(z)
    assert np.allclose(out, z)


def test_pava_makes_nondecreasing():
    z = np.array([0.0, 3.0, 1.0, 2.0, 5.0])  # 3→1 нарушение
    out = pava_increasing(z)
    assert np.all(np.diff(out) >= -1e-12)
    # PAVA сохраняет среднее (взвешенное) на блоках-нарушителях
    assert np.isclose(out.sum(), z.sum())


def test_pava_pools_violators_to_mean():
    z = np.array([5.0, 1.0])  # убывание -> оба сливаются в среднее 3
    out = pava_increasing(z)
    assert np.allclose(out, [3.0, 3.0])


# ─────────────────────────────────────────────────────────────────────────────
# monotone_despike — удаление «резких пиков» из монотонной величины
# ─────────────────────────────────────────────────────────────────────────────

def test_despike_decreasing_removes_spike_and_dip():
    x = np.linspace(0, 1, 20)
    true = 4000.0 - 3000.0 * x          # монотонно убывает (T по соплу)
    y = true.copy()
    y[8] += 600.0                       # одиночный пик вверх
    y[13] -= 400.0                      # одиночный провал вниз
    out = monotone_despike(y, decreasing=True)
    # выбросы устранены почти точно
    assert abs(out[8] - true[8]) < 60.0
    assert abs(out[13] - true[13]) < 60.0
    # пилообразность резко снижена
    assert _peakiness(out) < 0.3 * _peakiness(y)


def test_despike_increasing_velocity():
    x = np.linspace(0, 1, 24)
    true = 100.0 + 2500.0 * x ** 0.6    # монотонно растёт (V по соплу)
    y = true.copy()
    y[6] += 400.0
    y[15] -= 350.0
    out = monotone_despike(y, decreasing=False)
    assert abs(out[6] - true[6]) < 60.0
    assert abs(out[15] - true[15]) < 60.0


def test_despike_preserves_clean_monotone_profile():
    x = np.linspace(0, 1, 30)
    true = 13.4e6 * np.exp(-3.2 * x)    # гладкий монотонный профиль давления
    out = monotone_despike(true, decreasing=True)
    # на чистых данных правка минимальна (< 0.5 % размаха)
    assert np.max(np.abs(out - true)) < 0.005 * np.ptp(true)


def test_despike_handles_chamber_flat_then_expansion():
    # камера (плоский участок) + расширение в сопле — типичный профиль T
    flat = np.full(8, 4097.0)
    expand = 4097.0 * np.exp(-1.4 * np.linspace(0, 1, 24))
    true = np.concatenate([flat, expand])
    y = true.copy()
    y[10] += 350.0       # пик в сверхзвуковой части
    y[20] -= 300.0
    out = monotone_despike(y, decreasing=True)
    assert _peakiness(out) <= _peakiness(true) + 1e-6
    # монотонность не нарушена (плоский участок камеры допустим)
    assert np.all(np.diff(out) <= 1e-6)


def test_despike_removes_adjacent_spike_cluster():
    # ДВА СОСЕДНИХ выброса подряд — «зубцовый» тест бессилен (у каждого пика
    # сосед тоже испорчен), но критерий кривизны (2-я разность) их ловит.
    x = np.linspace(0, 1, 40)
    true = 4000.0 - 3000.0 * x          # монотонно убывает
    y = true.copy()
    y[18] -= 450.0                      # провал ↓ ...
    y[19] -= 380.0                      # ... и сразу второй рядом
    out = monotone_despike(y, decreasing=True)
    # «резкие пики» снесены до уровня чистого профиля
    assert _peakiness(out) <= 3.0 * _peakiness(true) + 1e-6
    # кластер выровнен по тренду
    assert abs(out[18] - true[18]) < 120.0
    assert abs(out[19] - true[19]) < 120.0


def test_despike_removes_triple_dip_cluster():
    # ТРИ испорченных точки подряд (экстремальный случай) — критерий кривизны
    # + финальная изотоническая проекция убирают остаточный пик.
    x = np.linspace(0, 1, 50)
    true = 300.0 * np.exp(-2.0 * x) + 50.0
    y = true.copy()
    y[18] -= 31.0
    y[19] -= 29.5
    y[20] -= 46.0
    out = monotone_despike(y, decreasing=True)
    # после чистки нет «резких пиков»: пиковость близка к идеально гладкой
    assert _peakiness(out) < 0.02


def test_despike_curvature_catches_steep_trend_spike():
    # На крутом монотонном тренде локальный «зуб» имеет огромную 2-ю разность,
    # хотя по абсолютной величине лежит близко к изотонической оценке.
    x = np.linspace(0, 1, 30)
    true = 13.4e6 * np.exp(-3.2 * x)    # очень крутое падение давления
    y = true.copy()
    y[5] *= 1.18                        # +18 % пик на крутом участке
    out = monotone_despike(y, decreasing=True)
    assert _peakiness(out) < 0.5 * _peakiness(y)


def test_despike_short_array_noop():
    y = np.array([3.0, 2.0, 1.0])  # n < 4
    out = monotone_despike(y, decreasing=True)
    assert np.allclose(out, y)


def test_despike_does_not_mutate_input():
    y = np.array([10.0, 9.0, 50.0, 7.0, 6.0, 5.0, 4.0])
    y_copy = y.copy()
    _ = monotone_despike(y, decreasing=True)
    assert np.allclose(y, y_copy)


def test_despike_with_nans_is_safe():
    y = np.array([10.0, np.nan, 8.0, 30.0, 6.0, 5.0, 4.0, 3.0])
    out = monotone_despike(y, decreasing=True)
    assert out.shape == y.shape  # не падает на NaN


# ─────────────────────────────────────────────────────────────────────────────
# hampel_filter — для немонотонных гладких величин
# ─────────────────────────────────────────────────────────────────────────────

def test_hampel_removes_isolated_spike():
    # лёгкая «дрожь» вокруг 1.20 + один явный выброс: MAD > 0, выброс ловится
    y = np.array([1.19, 1.21, 1.20, 2.50, 1.20, 1.21, 1.19, 1.20, 1.21])
    out = hampel_filter(y, window=2, n_sigma=3.0)
    assert abs(out[3] - 1.20) < 0.05


def test_hampel_short_array_noop():
    y = np.array([1.0, 5.0, 1.0])  # n < 2*window+1
    out = hampel_filter(y, window=2)
    assert np.allclose(out, y)


def test_hampel_does_not_mutate_input():
    y = np.array([1.0, 1.0, 9.0, 1.0, 1.0])
    y_copy = y.copy()
    _ = hampel_filter(y, window=2, n_sigma=2.0)
    assert np.allclose(y, y_copy)

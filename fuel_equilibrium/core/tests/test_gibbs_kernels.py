"""
Тесты нативных ядер целевой функции G/RT и градиента.

Проверяется, что оптимизированные ядра ``_gibbs_kernel`` / ``_grad_kernel``
(нативная компиляция через Numba, либо чистый Python при её отсутствии)
численно совпадают с эталонной реализацией «в лоб» (цикл на Python),
повторяющей исходный код решателя до оптимизации.

Это защищает от регрессий: любое изменение ядер, меняющее значение G/RT
или градиента, будет немедленно поймано.
"""

import math

import numpy as np
import pytest

from fuel_equilibrium.core.gibbs_solver import (
    _gibbs_kernel,
    _grad_kernel,
    _N_MIN,
    _HAVE_NUMBA,
)


# ── эталонные реализации (исходный чистый Python до оптимизации) ────────────
def _gibbs_ref(n, Ng, Nc, g0, ln_P, n_min=_N_MIN):
    G = 0.0
    ntot = max(n[:Ng].sum(), n_min)
    ln_nt = math.log(ntot)
    for i in range(Ng):
        ni = max(n[i], n_min)
        G += ni * (g0[i] + math.log(ni) - ln_nt + ln_P)
    for j in range(Nc):
        G += max(n[Ng + j], 0.0) * g0[Ng + j]
    return G


def _grad_ref(n, Ng, Nc, g0, ln_P, n_min=_N_MIN):
    N = Ng + Nc
    gr = np.zeros(N)
    ntot = max(n[:Ng].sum(), n_min)
    ln_nt = math.log(ntot)
    for i in range(Ng):
        gr[i] = g0[i] + math.log(max(n[i], n_min)) - ln_nt + ln_P
    for j in range(Nc):
        gr[Ng + j] = g0[Ng + j]
    return gr


def _call_gibbs(n, Ng, Nc, g0, ln_P):
    n = np.ascontiguousarray(n, dtype=np.float64)
    g0 = np.ascontiguousarray(g0, dtype=np.float64)
    return float(_gibbs_kernel(n, Ng, Nc, g0, ln_P, _N_MIN))


def _call_grad(n, Ng, Nc, g0, ln_P):
    n = np.ascontiguousarray(n, dtype=np.float64)
    g0 = np.ascontiguousarray(g0, dtype=np.float64)
    out = np.empty(Ng + Nc, dtype=np.float64)
    _grad_kernel(n, Ng, Nc, g0, ln_P, _N_MIN, out)
    return out


def test_have_numba_flag_is_bool():
    assert isinstance(_HAVE_NUMBA, bool)


def test_gibbs_kernel_matches_reference_simple():
    Ng, Nc = 4, 1
    g0 = np.array([-10.0, 5.0, 2.5, -3.3, 1.1])
    n = np.array([0.5, 1.2, 0.01, 0.3, 0.2])
    ln_P = 2.3
    got = _call_gibbs(n, Ng, Nc, g0, ln_P)
    ref = _gibbs_ref(n, Ng, Nc, g0, ln_P)
    assert abs(got - ref) <= 1e-9 * max(abs(ref), 1.0)


def test_grad_kernel_matches_reference_simple():
    Ng, Nc = 4, 1
    g0 = np.array([-10.0, 5.0, 2.5, -3.3, 1.1])
    n = np.array([0.5, 1.2, 0.01, 0.3, 0.2])
    ln_P = 2.3
    got = _call_grad(n, Ng, Nc, g0, ln_P)
    ref = _grad_ref(n, Ng, Nc, g0, ln_P)
    assert np.allclose(got, ref, atol=1e-12, rtol=0)


def test_kernels_handle_zero_and_tiny_moles():
    """Клиппинг n_min: нулевые и «следовые» моли не должны давать ln(0)/NaN."""
    Ng, Nc = 5, 2
    g0 = np.array([-8.0, 3.0, 0.0, -1.0, 4.0, 2.0, -2.0])
    n = np.array([0.0, 1e-25, 0.7, 0.0, 0.3, 1e-30, 0.1])
    ln_P = -1.5
    g = _call_gibbs(n, Ng, Nc, g0, ln_P)
    gr = _call_grad(n, Ng, Nc, g0, ln_P)
    assert math.isfinite(g)
    assert np.all(np.isfinite(gr))
    assert abs(g - _gibbs_ref(n, Ng, Nc, g0, ln_P)) <= 1e-9 * max(abs(g), 1.0)
    assert np.allclose(gr, _grad_ref(n, Ng, Nc, g0, ln_P), atol=1e-12, rtol=0)


def test_kernels_no_condensed():
    """Случай без конденсированной фазы (Nc = 0)."""
    Ng, Nc = 6, 0
    rng = np.random.default_rng(7)
    g0 = rng.uniform(-30, 30, Ng)
    n = np.abs(rng.uniform(0, 3, Ng))
    ln_P = 0.7
    assert abs(_call_gibbs(n, Ng, Nc, g0, ln_P)
               - _gibbs_ref(n, Ng, Nc, g0, ln_P)) <= 1e-9
    assert np.allclose(_call_grad(n, Ng, Nc, g0, ln_P),
                       _grad_ref(n, Ng, Nc, g0, ln_P), atol=1e-12, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2, 13, 99])
def test_kernels_match_reference_randomized(seed):
    """Случайные конфигурации, включая разное число видов и фаз."""
    rng = np.random.default_rng(seed)
    Ng = int(rng.integers(3, 60))
    Nc = int(rng.integers(0, 4))
    N = Ng + Nc
    g0 = rng.uniform(-50, 50, N)
    ln_P = float(rng.uniform(-5, 5))
    n = np.abs(rng.uniform(0, 5, N))
    # вкрапим нулевые/следовые значения
    n[rng.integers(0, N)] = 0.0
    n[rng.integers(0, N)] = 1e-25

    g_got = _call_gibbs(n, Ng, Nc, g0, ln_P)
    g_ref = _gibbs_ref(n, Ng, Nc, g0, ln_P)
    assert abs(g_got - g_ref) <= 1e-9 * max(abs(g_ref), 1.0)

    gr_got = _call_grad(n, Ng, Nc, g0, ln_P)
    gr_ref = _grad_ref(n, Ng, Nc, g0, ln_P)
    assert np.allclose(gr_got, gr_ref, atol=1e-11, rtol=0)

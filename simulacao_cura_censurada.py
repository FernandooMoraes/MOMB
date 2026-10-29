"""Simulacao Monte Carlo reproduzivel para a MOMB defeituosa.

Parametrizacao usada na otimizacao
----------------------------------
    k = -a = exp(eta_k) > 0
    p = expit(eta_p) in (0, 1)
    b = p / (2 * (p - 1)) < 0

Se F_MB(t; k) for a CDF Maxwell usual com escala k, a MOMB defeituosa pode
ser escrita de forma numericamente estavel como

    S(t; k, p) = p(1 + F_MB) / [p + (2-p)F_MB]
    f(t; k, p) = 2p(1-p)f_MB / [p + (2-p)F_MB]^2..
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable

import joblib

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "momb_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy.integrate import quad
from scipy.optimize import approx_fprime, brentq, minimize
from scipy.special import expit, logit
from scipy.stats import binom, binomtest, maxwell, norm


Z_95 = float(norm.ppf(0.975))
UNIT_MAXWELL_MEDIAN = float(maxwell.ppf(0.5))
EPS_PROB = 1e-12


@dataclass(frozen=True)
class Scenario:
    true_a: float
    true_p: float
    n: int
    susceptible_censor_target: float
    mechanism: str = "random_uniform"
    censoring_parameter: float = math.nan

    @property
    def true_b(self) -> float:
        return b_from_p(self.true_p)

    @property
    def expected_total_censoring(self) -> float:
        return self.true_p + (1.0 - self.true_p) * self.susceptible_censor_target


def b_from_p(p: float | np.ndarray) -> float | np.ndarray:
    """Converte fracao de cura p em b < 0."""
    return -np.asarray(p) / (2.0 * (1.0 - np.asarray(p)))


def p_from_b(b: float | np.ndarray) -> float | np.ndarray:
    """Converte b < 0 em fracao de cura p."""
    return 2.0 * np.asarray(b) / (2.0 * np.asarray(b) - 1.0)


def momb_survival(t: np.ndarray | float, a: float, p: float) -> np.ndarray:
    """Sobrevivencia defeituosa MOMB em parametrizacao (a, p)."""
    k = -float(a)
    if not (k > 0.0 and 0.0 < p < 1.0):
        raise ValueError("A MOMB defeituosa requer a < 0 e 0 < p < 1.")
    f_mb = maxwell.cdf(np.asarray(t, dtype=float), scale=k)
    denominator = p + (2.0 - p) * f_mb
    return p * (1.0 + f_mb) / denominator


def momb_density(t: np.ndarray | float, a: float, p: float) -> np.ndarray:
    """Densidade impropria MOMB; sua integral e 1-p."""
    k = -float(a)
    if not (k > 0.0 and 0.0 < p < 1.0):
        raise ValueError("A MOMB defeituosa requer a < 0 e 0 < p < 1.")
    values = np.asarray(t, dtype=float)
    f_mb = maxwell.cdf(values, scale=k)
    density_mb = maxwell.pdf(values, scale=k)
    denominator = p + (2.0 - p) * f_mb
    return 2.0 * p * (1.0 - p) * density_mb / denominator**2


def survival_level_to_maxwell_cdf(survival_level: np.ndarray | float, p: float) -> np.ndarray:
    """Inverte algebricamente S_MOMB(t)=q para obter F_MB(t)."""
    q = np.asarray(survival_level, dtype=float)
    denominator = q * (2.0 - p) - p
    f_mb = p * (1.0 - q) / denominator
    return np.clip(f_mb, EPS_PROB, 1.0 - EPS_PROB)


def susceptible_times(rng: np.random.Generator, size: int, a: float, p: float) -> np.ndarray:
    """Gera tempos T | individuo suscetivel por inversao sem busca numerica."""
    if size == 0:
        return np.empty(0, dtype=float)
    conditional_survival = rng.uniform(EPS_PROB, 1.0 - EPS_PROB, size=size)
    defective_survival = p + (1.0 - p) * conditional_survival
    f_mb = survival_level_to_maxwell_cdf(defective_survival, p)
    return maxwell.ppf(f_mb, scale=-a)


def administrative_cutoff(a: float, p: float, susceptible_censor_target: float) -> float:
    """Cutoff tau tal que P(T > tau | suscetivel) e o alvo informado."""
    q = p + (1.0 - p) * susceptible_censor_target
    f_mb = survival_level_to_maxwell_cdf(q, p)
    return float(maxwell.ppf(f_mb, scale=-a))


def conditional_survival(t: np.ndarray | float, a: float, p: float) -> np.ndarray:
    """Sobrevivência condicional ao indivíduo ser suscetível."""
    return (momb_survival(t, a, p) - p) / (1.0 - p)


@lru_cache(maxsize=None)
def uniform_censoring_cutoff(a: float, p: float, susceptible_censor_target: float) -> float:
    """Calibra tau para C~Uniforme(0,tau) dar a censura suscetível desejada.

    Para um censor independente C, P(C<T|susc) = E[min(T/tau,1)]
    = integral_0^tau S_susc(t) dt / tau.
    """
    target = float(susceptible_censor_target)
    if not (0.0 <= target < 1.0):
        raise ValueError("O alvo de censura deve estar em [0,1).")
    if target == 0.0:
        return float("inf")

    scale = max(-float(a), 1e-8)

    def achieved(tau: float) -> float:
        integral = quad(
            lambda value: float(conditional_survival(value, a, p)),
            0.0,
            tau,
            epsabs=1e-9,
            epsrel=1e-9,
            limit=150,
        )[0]
        return integral / tau

    def equation(log_tau: float) -> float:
        tau = scale * math.exp(log_tau)
        return achieved(tau) - target

    lower, upper = -12.0, 4.0
    while equation(upper) > 0.0 and upper < 40.0:
        upper += 2.0
    root = brentq(equation, lower, upper, xtol=1e-10, rtol=1e-10)
    return float(scale * math.exp(root))


def simulate_cohort(
    rng: np.random.Generator,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Simula cura, falha latente e censura independente ou administrativa."""
    cured = rng.random(scenario.n) < scenario.true_p
    latent_time = np.full(scenario.n, np.inf, dtype=float)
    susceptible = ~cured
    latent_time[susceptible] = susceptible_times(
        rng,
        int(susceptible.sum()),
        scenario.true_a,
        scenario.true_p,
    )

    if scenario.mechanism == "random_uniform":
        tau = scenario.censoring_parameter
        if not np.isfinite(tau):
            tau = uniform_censoring_cutoff(
                scenario.true_a, scenario.true_p, scenario.susceptible_censor_target
            )
        censor_time = rng.uniform(0.0, tau, size=scenario.n)
    elif scenario.mechanism == "administrative":
        tau = scenario.censoring_parameter
        if not np.isfinite(tau):
            tau = administrative_cutoff(
                scenario.true_a,
                scenario.true_p,
                scenario.susceptible_censor_target,
            )
        censor_time = np.full(scenario.n, tau, dtype=float)
    else:
        raise ValueError(f"Mecanismo desconhecido: {scenario.mechanism}")
    event = susceptible & (latent_time <= censor_time)
    observed_time = np.minimum(latent_time, censor_time)
    delta = event.astype(np.int8)

    diagnostics: dict[str, float | int] = {
        "n_cured": int(cured.sum()),
        "n_susceptible": int(susceptible.sum()),
        "n_events": int(event.sum()),
        "n_censored": int((1 - delta).sum()),
        "realized_cure_fraction": float(cured.mean()),
        "realized_censoring_fraction": float((1 - delta).mean()),
        "censoring_parameter_used": float(tau),
    }
    return observed_time, delta, diagnostics


def unpack_eta(eta: np.ndarray) -> tuple[float, float, float, float]:
    """Retorna (k, a, p, b) a partir dos parametros irrestritos."""
    k = float(np.exp(eta[0]))
    p = float(expit(eta[1]))
    a = -k
    b = float(b_from_p(np.clip(p, EPS_PROB, 1.0 - EPS_PROB)))
    return k, a, p, b


def nll_and_gradient(
    eta: np.ndarray,
    observed_time: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Log-verossimilhanca negativa censurada e gradiente analitico."""
    k, _, p, _ = unpack_eta(eta)
    if not (np.isfinite(k) and 1e-10 < p < 1.0 - 1e-10):
        return 1e100, np.zeros(2, dtype=float)

    t = observed_time
    f_mb = maxwell.cdf(t, scale=k)
    log_density_mb = maxwell.logpdf(t, scale=k)
    denominator = p + (2.0 - p) * f_mb
    if np.any(denominator <= 0.0) or np.any(~np.isfinite(log_density_mb)):
        return 1e100, np.zeros(2, dtype=float)

    log_survival = np.log(p) + np.log1p(f_mb) - np.log(denominator)
    log_density = (
        math.log(2.0)
        + np.log(p)
        + np.log1p(-p)
        + log_density_mb
        - 2.0 * np.log(denominator)
    )
    loglik = np.sum(delta * log_density + (1 - delta) * log_survival)

    density_mb = np.exp(log_density_mb)
    derivative_f_eta_k = -t * density_mb
    derivative_logpdf_eta_k = -3.0 + (t / k) ** 2
    derivative_denominator_eta_k = (2.0 - p) * derivative_f_eta_k

    score_survival_eta_k = (
        derivative_f_eta_k / (1.0 + f_mb)
        - derivative_denominator_eta_k / denominator
    )
    score_density_eta_k = (
        derivative_logpdf_eta_k
        - 2.0 * derivative_denominator_eta_k / denominator
    )

    derivative_denominator_p = 1.0 - f_mb
    score_survival_p = 1.0 / p - derivative_denominator_p / denominator
    score_density_p = (
        1.0 / p
        - 1.0 / (1.0 - p)
        - 2.0 * derivative_denominator_p / denominator
    )
    derivative_p_eta_p = p * (1.0 - p)
    score_survival_eta_p = score_survival_p * derivative_p_eta_p
    score_density_eta_p = score_density_p * derivative_p_eta_p

    score_eta_k = np.sum(delta * score_density_eta_k + (1 - delta) * score_survival_eta_k)
    score_eta_p = np.sum(delta * score_density_eta_p + (1 - delta) * score_survival_eta_p)
    gradient = -np.array([score_eta_k, score_eta_p], dtype=float)
    return float(-loglik), gradient


def observed_hessian(
    eta: np.ndarray,
    observed_time: np.ndarray,
    delta: np.ndarray,
    relative_step: float = 2e-4,
) -> np.ndarray:
    """Hessiana observada por diferencas centrais do gradiente analitico."""
    dimension = len(eta)
    hessian = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        step = relative_step * max(1.0, abs(float(eta[column])))
        plus = eta.copy()
        minus = eta.copy()
        plus[column] += step
        minus[column] -= step
        gradient_plus = nll_and_gradient(plus, observed_time, delta)[1]
        gradient_minus = nll_and_gradient(minus, observed_time, delta)[1]
        hessian[:, column] = (gradient_plus - gradient_minus) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def initial_points(
    observed_time: np.ndarray,
    delta: np.ndarray,
    start_profile: str = "full",
) -> list[np.ndarray]:
    """Pontos iniciais determinados apenas pelos dados, nunca pelo parametro verdadeiro."""
    failures = observed_time[delta == 1]
    reference = float(np.median(failures if len(failures) >= 3 else observed_time))
    scale_base = max(reference / UNIT_MAXWELL_MEDIAN, 1e-4)
    if start_profile == "full":
        p_grid = (0.08, 0.25, 0.50, 0.75)
        scale_multipliers = (0.65, 1.35)
    elif start_profile == "reduced":
        p_grid = (0.15, 0.60)
        scale_multipliers = (0.70, 1.40)
    else:
        raise ValueError(f"Perfil de iniciais desconhecido: {start_profile}")
    return [
        np.array([np.log(scale_base * multiplier), logit(p_start)], dtype=float)
        for p_start, multiplier in product(p_grid, scale_multipliers)
    ]


def fit_momb(
    observed_time: np.ndarray,
    delta: np.ndarray,
    maxiter: int,
    gtol: float,
    start_profile: str = "full",
) -> dict[str, float | int | bool | str]:
    """Ajusta por multiplos inicios e valida gradiente e informacao observada."""
    starts = initial_points(observed_time, delta, start_profile=start_profile)
    candidates = []
    for start_index, start in enumerate(starts):
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            result = minimize(
                nll_and_gradient,
                start,
                args=(observed_time, delta),
                method="BFGS",
                jac=True,
                options={"gtol": gtol, "maxiter": maxiter, "disp": False},
            )
        warning_messages = sorted({str(item.message) for item in captured_warnings})
        if np.isfinite(result.fun):
            candidates.append((float(result.fun), start_index, result, warning_messages))

    if not candidates:
        return {"fit_valid": False, "failure_reason": "no_finite_optimizer_candidate"}

    _, selected_start, best, best_warnings = min(candidates, key=lambda item: item[0])
    eta_hat = np.asarray(best.x, dtype=float)
    nll, gradient = nll_and_gradient(eta_hat, observed_time, delta)
    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))

    base: dict[str, float | int | bool | str] = {
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        # Mensagens emitidas pela line search podem depender do escalonamento
        # das threads. Registramos apenas um marcador determinístico derivado
        # do resultado final, preservando reprodutibilidade byte a byte.
        "optimizer_warning_count": int(not bool(best.success)),
        "optimizer_warnings": "optimizer_not_successful" if not best.success else "",
        "selected_start": int(selected_start),
        "n_starts": len(starts),
        "optimizer_iterations": int(getattr(best, "nit", -1)),
        "optimizer_function_evaluations": int(getattr(best, "nfev", -1)),
        "negative_loglik": nll,
        "gradient_inf_norm": gradient_norm,
        "eta_k_hat": float(eta_hat[0]),
        "eta_p_hat": float(eta_hat[1]),
    }

    try:
        hessian = observed_hessian(eta_hat, observed_time, delta)
        eigenvalues = np.linalg.eigvalsh(hessian)
        hessian_condition = float(np.linalg.cond(hessian))
        base.update(
            {
                "hessian_min_eigenvalue": float(eigenvalues.min()),
                "hessian_max_eigenvalue": float(eigenvalues.max()),
                "hessian_condition_number": hessian_condition,
            }
        )
        information_ok = bool(
            np.all(np.isfinite(eigenvalues))
            and eigenvalues.min() > 1e-8
            and np.isfinite(hessian_condition)
            and hessian_condition < 1e10
        )
        if not information_ok:
            base.update({"fit_valid": False, "failure_reason": "invalid_observed_information"})
            return base
        covariance_eta = np.linalg.inv(hessian)
        covariance_diagonal = np.diag(covariance_eta)
        if np.any(covariance_diagonal <= 0.0) or np.any(~np.isfinite(covariance_diagonal)):
            base.update({"fit_valid": False, "failure_reason": "invalid_covariance_diagonal"})
            return base
        se_eta = np.sqrt(covariance_diagonal)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        base.update({"fit_valid": False, "failure_reason": f"hessian_error:{type(exc).__name__}"})
        return base

    optimizer_ok = bool(best.success or gradient_norm <= 5e-5)
    gradient_ok = bool(np.isfinite(gradient_norm) and gradient_norm <= 5e-4)
    k_hat, a_hat, p_hat, b_hat = unpack_eta(eta_hat)
    interior_ok = bool(1e-6 < p_hat < 1.0 - 1e-6 and np.isfinite(k_hat))
    se_ok = bool(np.all(np.isfinite(se_eta)) and np.all((se_eta > 0.0) & (se_eta < 50.0)))
    if not optimizer_ok:
        reason = "optimizer_not_converged"
    elif not gradient_ok:
        reason = "gradient_too_large"
    elif not information_ok:
        reason = "invalid_observed_information"
    elif not interior_ok:
        reason = "estimate_on_practical_boundary"
    elif not se_ok:
        reason = "invalid_standard_error"
    else:
        reason = ""

    se_eta_k, se_eta_p = map(float, se_eta)
    se_a = abs(a_hat) * se_eta_k
    se_b = abs(b_hat) * se_eta_p
    se_p = p_hat * (1.0 - p_hat) * se_eta_p

    eta_k_low, eta_k_high = eta_hat[0] - Z_95 * se_eta_k, eta_hat[0] + Z_95 * se_eta_k
    eta_p_low, eta_p_high = eta_hat[1] - Z_95 * se_eta_p, eta_hat[1] + Z_95 * se_eta_p
    a_ci_lower = -np.exp(np.clip(eta_k_high, -700.0, 700.0))
    a_ci_upper = -np.exp(np.clip(eta_k_low, -700.0, 700.0))
    p_ci_lower = float(np.clip(expit(eta_p_low), EPS_PROB, 1.0 - EPS_PROB))
    p_ci_upper = float(np.clip(expit(eta_p_high), EPS_PROB, 1.0 - EPS_PROB))
    b_ci_lower, b_ci_upper = b_from_p(p_ci_upper), b_from_p(p_ci_lower)

    base.update(
        {
            "fit_valid": reason == "",
            "failure_reason": reason,
            "k_hat": k_hat,
            "a_hat": a_hat,
            "b_hat": b_hat,
            "p_hat": p_hat,
            "se_eta_k": se_eta_k,
            "se_eta_p": se_eta_p,
            "se_a": se_a,
            "se_b": se_b,
            "se_p": se_p,
            "a_ci_lower": float(a_ci_lower),
            "a_ci_upper": float(a_ci_upper),
            "b_ci_lower": float(b_ci_lower),
            "b_ci_upper": float(b_ci_upper),
            "p_ci_lower": float(p_ci_lower),
            "p_ci_upper": float(p_ci_upper),
        }
    )
    return base


def run_replication(
    scenario: Scenario,
    replication: int,
    seed: int,
    maxiter: int,
    gtol: float,
    start_profile: str = "full",
) -> dict[str, float | int | bool | str]:
    """Executa uma replicacao; qualquer falha vira um registro auditavel."""
    base: dict[str, float | int | bool | str] = {
        **asdict(scenario),
        "true_b": scenario.true_b,
        "expected_total_censoring": scenario.expected_total_censoring,
        "replication": replication,
        "replication_seed": seed,
    }
    try:
        rng = np.random.default_rng(seed)
        observed_time, delta, generation = simulate_cohort(rng, scenario)
        base.update(generation)
        if int(delta.sum()) < 5:
            base.update({"fit_valid": False, "failure_reason": "fewer_than_five_events"})
            return base
        fit = fit_momb(
            observed_time,
            delta,
            maxiter=maxiter,
            gtol=gtol,
            start_profile=start_profile,
        )
        base.update(fit)
    except Exception as exc:  # Mantem a taxa de falha como parte do experimento.
        base.update(
            {
                "fit_valid": False,
                "failure_reason": f"unexpected:{type(exc).__name__}",
                "exception_message": str(exc),
            }
        )
    return base


def add_performance_columns(results: pd.DataFrame) -> pd.DataFrame:
    enriched = results.copy()
    for parameter in ("a", "b", "p"):
        estimate = f"{parameter}_hat"
        true_value = f"true_{parameter}"
        enriched[f"bias_{parameter}"] = enriched[estimate] - enriched[true_value]
        enriched[f"squared_error_{parameter}"] = enriched[f"bias_{parameter}"] ** 2
        enriched[f"cover_{parameter}"] = (
            (enriched[true_value] >= enriched[f"{parameter}_ci_lower"])
            & (enriched[true_value] <= enriched[f"{parameter}_ci_upper"])
        )
        enriched[f"ci_width_{parameter}"] = (
            enriched[f"{parameter}_ci_upper"] - enriched[f"{parameter}_ci_lower"]
        )
    return enriched


def exact_nominal_coverage_range(
    replications: int,
    nominal_coverage: float = 0.95,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Faixa preditiva binomial exata para a cobertura Monte Carlo observada."""
    alpha = 1.0 - confidence_level
    lower_count = int(binom.ppf(alpha / 2.0, replications, nominal_coverage))
    upper_count = int(binom.ppf(1.0 - alpha / 2.0, replications, nominal_coverage))
    return lower_count / replications, upper_count / replications


def wilson_interval(successes: int, replications: int) -> tuple[float, float]:
    """Intervalo de Wilson de 95% para a cobertura verdadeira de um ponto."""
    proportion = successes / replications
    denominator = 1.0 + Z_95**2 / replications
    center = (proportion + Z_95**2 / (2.0 * replications)) / denominator
    half_width = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / replications
            + Z_95**2 / (4.0 * replications**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "true_a",
        "true_b",
        "true_p",
        "n",
        "susceptible_censor_target",
        "mechanism",
        "censoring_parameter",
        "expected_total_censoring",
    ]
    quality = (
        results.groupby(group_columns, dropna=False)
        .agg(
            attempted_replications=("replication", "size"),
            valid_replications=("fit_valid", "sum"),
            fit_valid_rate=("fit_valid", "mean"),
            optimizer_success_rate=("optimizer_success", "mean"),
            mean_events=("n_events", "mean"),
            mean_realized_cure_fraction=("realized_cure_fraction", "mean"),
            mean_realized_censoring_fraction=("realized_censoring_fraction", "mean"),
            median_gradient_inf_norm=("gradient_inf_norm", "median"),
            median_hessian_condition_number=("hessian_condition_number", "median"),
        )
        .reset_index()
    )

    valid = results[results["fit_valid"]].copy()
    summary = quality
    for parameter in ("a", "b", "p"):
        parameter_summary = (
            valid.groupby(group_columns, dropna=False)
            .agg(
                **{
                    f"mean_estimate_{parameter}": (f"{parameter}_hat", "mean"),
                    f"bias_{parameter}": (f"bias_{parameter}", "mean"),
                    f"mse_{parameter}": (f"squared_error_{parameter}", "mean"),
                    f"empirical_sd_{parameter}": (f"{parameter}_hat", "std"),
                    f"mean_se_{parameter}": (f"se_{parameter}", "mean"),
                    f"median_se_{parameter}": (f"se_{parameter}", "median"),
                    f"coverage_{parameter}": (f"cover_{parameter}", "mean"),
                    f"covered_count_{parameter}": (f"cover_{parameter}", "sum"),
                    f"mean_ci_width_{parameter}": (f"ci_width_{parameter}", "mean"),
                }
            )
            .reset_index()
        )
        parameter_summary[f"rmse_{parameter}"] = np.sqrt(parameter_summary[f"mse_{parameter}"])
        parameter_summary[f"relative_bias_{parameter}"] = (
            parameter_summary[f"bias_{parameter}"] / parameter_summary[f"true_{parameter}"]
        )
        parameter_summary[f"mean_se_to_empirical_sd_{parameter}"] = (
            parameter_summary[f"mean_se_{parameter}"]
            / parameter_summary[f"empirical_sd_{parameter}"]
        )
        summary = summary.merge(parameter_summary, on=group_columns, how="left")

    valid_count = summary["valid_replications"].clip(lower=1)
    for parameter in ("a", "b", "p"):
        summary[f"coverage_mcse_{parameter}"] = np.sqrt(
            summary[f"coverage_{parameter}"]
            * (1.0 - summary[f"coverage_{parameter}"])
            / valid_count
        )
        expected_ranges = [
            exact_nominal_coverage_range(int(replications)) for replications in valid_count
        ]
        summary[f"expected_coverage_lower_{parameter}"] = [item[0] for item in expected_ranges]
        summary[f"expected_coverage_upper_{parameter}"] = [item[1] for item in expected_ranges]
        wilson_ranges = [
            wilson_interval(int(successes), int(replications))
            for successes, replications in zip(
                summary[f"covered_count_{parameter}"], valid_count
            )
        ]
        summary[f"coverage_wilson_lower_{parameter}"] = [item[0] for item in wilson_ranges]
        summary[f"coverage_wilson_upper_{parameter}"] = [item[1] for item in wilson_ranges]
        summary[f"coverage_compatible_with_95_{parameter}"] = (
            (summary[f"coverage_{parameter}"] >= summary[f"expected_coverage_lower_{parameter}"])
            & (summary[f"coverage_{parameter}"] <= summary[f"expected_coverage_upper_{parameter}"])
        )
        summary[f"coverage_exact_binomial_pvalue_{parameter}"] = [
            binomtest(int(successes), int(replications), 0.95).pvalue
            for successes, replications in zip(
                summary[f"covered_count_{parameter}"], valid_count
            )
        ]
    return summary.sort_values(group_columns).reset_index(drop=True)


def coverage_assessment_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Tabela longa para auditoria especifica da cobertura em cada ponto."""
    records: list[dict[str, float | int | bool | str]] = []
    identity_columns = [
        "true_a", "true_b", "true_p", "n", "susceptible_censor_target",
        "mechanism", "censoring_parameter",
        "expected_total_censoring", "attempted_replications", "valid_replications",
    ]
    for _, row in summary.iterrows():
        identity = {column: row[column] for column in identity_columns}
        for parameter in ("a", "b", "p"):
            records.append(
                {
                    **identity,
                    "parameter": parameter,
                    "covered_count": int(row[f"covered_count_{parameter}"]),
                    "observed_coverage": row[f"coverage_{parameter}"],
                    "expected_coverage_lower_exact": row[f"expected_coverage_lower_{parameter}"],
                    "nominal_coverage": 0.95,
                    "expected_coverage_upper_exact": row[f"expected_coverage_upper_{parameter}"],
                    "observed_coverage_wilson_lower": row[f"coverage_wilson_lower_{parameter}"],
                    "observed_coverage_wilson_upper": row[f"coverage_wilson_upper_{parameter}"],
                    "exact_binomial_pvalue": row[f"coverage_exact_binomial_pvalue_{parameter}"],
                    "compatible_with_nominal_95": row[f"coverage_compatible_with_95_{parameter}"],
                    "coverage_equivalence_group": "eta_p" if parameter in ("b", "p") else "eta_k",
                }
            )
    return pd.DataFrame(records)


def plot_summary(summary: pd.DataFrame, output_directory: Path) -> list[str]:
    created: list[str] = []
    colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, max(2, summary["true_p"].nunique()))
    )
    p_colors = dict(zip(sorted(summary["true_p"].unique()), colors))
    for (mechanism, censor_target), panel in summary.groupby(
        ["mechanism", "susceptible_censor_target"]
    ):
        figure, axes = plt.subplots(3, 3, figsize=(15, 12), sharex="col")
        for true_a, linestyle in zip(sorted(panel["true_a"].unique()), ("-", "--", ":", "-.")):
            for true_p in sorted(panel["true_p"].unique()):
                scenario = panel[(panel["true_a"] == true_a) & (panel["true_p"] == true_p)]
                if scenario.empty:
                    continue
                true_b = float(scenario["true_b"].iloc[0])
                label = rf"$a={true_a:g}$, $b={true_b:.4g}$, $p={true_p:.1f}$"
                for column, parameter in enumerate(("a", "b", "p")):
                    axes[0, column].plot(
                        scenario["n"], scenario[f"bias_{parameter}"],
                        marker="o", color=p_colors[true_p], linestyle=linestyle, label=label,
                    )
                    axes[1, column].plot(
                        scenario["n"], scenario[f"rmse_{parameter}"],
                        marker="o", color=p_colors[true_p], linestyle=linestyle,
                    )
                    axes[2, column].plot(
                        scenario["n"], scenario[f"coverage_{parameter}"],
                        marker="o", color=p_colors[true_p], linestyle=linestyle,
                    )

        for column, parameter in enumerate(("a", "b", "p")):
            axes[0, column].axhline(0.0, color="black", linewidth=1)
            axes[1, column].axhline(0.0, color="black", linewidth=1)
            axes[2, column].axhline(0.95, color="black", linewidth=1, linestyle="--")
            axes[0, column].set_title({"a": "Parameter a", "b": "Parameter b", "p": "Cure fraction p"}[parameter])
            axes[2, column].set_xlabel("Cohort size (n)")
            axes[2, column].set_ylim(0.75, 1.01)
        axes[0, 0].set_ylabel("Bias")
        axes[1, 0].set_ylabel("RMSE")
        axes[2, 0].set_ylabel("Coverage (95%)")
        for axis in axes.flat:
            axis.grid(alpha=0.25, linestyle=":")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        figure.legend(
            unique.values(), unique.keys(), loc="lower center", ncol=4,
            frameon=True, bbox_to_anchor=(0.5, 0.005),
        )
        figure.suptitle(
            "MOMB censored-cohort Monte Carlo\n"
            f"Mechanism: {mechanism}; target censoring among susceptible subjects: {censor_target:.0%}",
            fontsize=16,
        )
        figure.tight_layout(rect=(0, 0.08, 1, 0.94))
        name = (
            f"performance_{mechanism}_censor_"
            f"{str(f'{censor_target:.2f}').replace('.', 'p')}.png"
        )
        figure.savefig(output_directory / name, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(name)
    return created


def plot_coverage_diagnostics(summary: pd.DataFrame, output_directory: Path) -> list[str]:
    """Cobertura com faixa binomial esperada e incerteza observada ponto a ponto."""
    created: list[str] = []
    for (mechanism, censor_target), panel in summary.groupby(
        ["mechanism", "susceptible_censor_target"]
    ):
        a_values = sorted(panel["true_a"].unique())
        p_values = sorted(panel["true_p"].unique())
        figure, axes = plt.subplots(
            len(a_values), len(p_values), figsize=(16, 7.5), sharey=True, squeeze=False
        )
        for row_index, true_a in enumerate(a_values):
            for column_index, true_p in enumerate(p_values):
                axis = axes[row_index, column_index]
                scenario = panel[
                    (panel["true_a"] == true_a) & (panel["true_p"] == true_p)
                ].sort_values("n")
                x = scenario["n"].to_numpy(dtype=float)
                expected_lower = scenario["expected_coverage_lower_a"].to_numpy(dtype=float)
                expected_upper = scenario["expected_coverage_upper_a"].to_numpy(dtype=float)
                axis.fill_between(
                    x,
                    expected_lower,
                    expected_upper,
                    color="#b7e4c7",
                    alpha=0.55,
                    label="Exact expected range under 95% coverage",
                )
                axis.plot(x, expected_lower, color="#40916c", linewidth=0.8, linestyle=":")
                axis.plot(x, expected_upper, color="#40916c", linewidth=0.8, linestyle=":")
                axis.axhline(0.95, color="black", linewidth=1.1, linestyle="--", label="Nominal 0.95")

                for parameter, color, marker, label in (
                    ("a", "#1f77b4", "o", "Parameter a"),
                    ("b", "#2ca02c", "^", "Parameter b"),
                    ("p", "#d95f02", "s", "Cure fraction p"),
                ):
                    observed = scenario[f"coverage_{parameter}"].to_numpy(dtype=float)
                    lower = scenario[f"coverage_wilson_lower_{parameter}"].to_numpy(dtype=float)
                    upper = scenario[f"coverage_wilson_upper_{parameter}"].to_numpy(dtype=float)
                    y_error = np.vstack(
                        (np.maximum(observed - lower, 0.0), np.maximum(upper - observed, 0.0))
                    )
                    axis.errorbar(
                        x,
                        observed,
                        yerr=y_error,
                        color=color,
                        marker=marker,
                        linewidth=1.5,
                        capsize=3,
                        label=label,
                    )

                valid_counts = scenario["valid_replications"].astype(int).to_numpy()
                axis.set_xticks(x)
                axis.set_xticklabels(
                    [str(int(n)) for n in x], fontsize=7, rotation=45, ha="right"
                )
                true_b = float(scenario["true_b"].iloc[0])
                axis.set_title(rf"$a={true_a:g}$, $b={true_b:.4g}$, $p={true_p:.1f}$")
                axis.set_ylim(0.82, 1.005)
                axis.grid(alpha=0.25, linestyle=":")
                if row_index == len(a_values) - 1:
                    axis.set_xlabel("Cohort size n (pointwise valid R in CSV)")
                if column_index == 0:
                    axis.set_ylabel("Coverage probability")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        figure.legend(
            unique.values(), unique.keys(), loc="lower center", ncol=4,
            frameon=True, bbox_to_anchor=(0.5, 0.005),
        )
        figure.suptitle(
            "MOMB coverage diagnostic\n"
            "Pointwise exact binomial expectation and Wilson uncertainty for observed coverage\n"
            f"Mechanism: {mechanism}; target censoring among susceptible subjects: {censor_target:.0%}",
            fontsize=15,
        )
        figure.tight_layout(rect=(0, 0.09, 1, 0.88))
        name = (
            f"coverage_diagnostic_{mechanism}_censor_"
            f"{str(f'{censor_target:.2f}').replace('.', 'p')}.png"
        )
        figure.savefig(output_directory / name, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(name)
    return created


def plot_quality_diagnostics(summary: pd.DataFrame, output_directory: Path) -> list[str]:
    """Gera diagnósticos de validade e calibração de erros-padrão."""
    created: list[str] = []
    for (mechanism, censor_target), panel in summary.groupby(
        ["mechanism", "susceptible_censor_target"]
    ):
        panel = panel.copy()
        panel["scenario_label"] = panel.apply(
            lambda row: f"a={row.true_a:g}, p={row.true_p:.1f}", axis=1
        )
        labels = sorted(panel["scenario_label"].unique())
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(labels))))
        figure, axes = plt.subplots(1, 4, figsize=(21, 5.5), sharex=True)
        for label, color in zip(labels, colors):
            scenario = panel[panel["scenario_label"] == label].sort_values("n")
            axes[0].plot(
                scenario["n"], scenario["fit_valid_rate"], marker="o", color=color, label=label
            )
            for axis, parameter in zip(axes[1:], ("a", "b", "p")):
                ratio = scenario[f"mean_se_to_empirical_sd_{parameter}"]
                axis.plot(scenario["n"], ratio, marker="o", color=color, label=label)
        axes[0].axhline(0.98, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Valid-fit rate")
        axes[0].set_ylim(0.0, 1.02)
        for axis, parameter in zip(axes[1:], ("a", "b", "p")):
            axis.axhspan(0.90, 1.10, color="#b7e4c7", alpha=0.5)
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel(f"Mean SE / empirical SD ({parameter})")
        for axis in axes:
            axis.set_xlabel("Cohort size n")
            axis.set_xticks(sorted(panel["n"].unique()))
            axis.grid(alpha=0.25, linestyle=":")
        figure.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
        figure.suptitle(
            f"MOMB diagnostics — {mechanism}; susceptible censoring target {censor_target:.0%}"
        )
        figure.tight_layout(rect=(0, 0.1, 1, 0.94))
        name = (
            f"quality_{mechanism}_censor_"
            f"{str(f'{censor_target:.2f}').replace('.', 'p')}.png"
        )
        figure.savefig(output_directory / name, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(name)
    return created


def write_article_tables(summary: pd.DataFrame, output_directory: Path) -> list[str]:
    created: list[str] = []
    main = summary[
        (summary["mechanism"] == "random_uniform")
        & np.isclose(summary["true_a"], -1.0)
        & np.isclose(summary["susceptible_censor_target"], 0.10)
        & summary["n"].isin([50, 250, 1000])
    ].copy()
    main_columns = [
        "true_a", "true_b", "true_p", "n", "susceptible_censor_target", "mechanism",
        "valid_replications", "fit_valid_rate", "bias_a", "rmse_a",
        "empirical_sd_a", "mean_se_a", "coverage_a", "bias_b", "rmse_b",
        "empirical_sd_b", "mean_se_b", "coverage_b", "bias_p", "rmse_p",
        "empirical_sd_p", "mean_se_p", "coverage_p", "mean_ci_width_p",
    ]
    main = main[[column for column in main_columns if column in main.columns]]
    main.to_csv(output_directory / "table_main.csv", index=False)
    created.append("table_main.csv")

    robustness = (
        summary.groupby(["mechanism", "n", "susceptible_censor_target"], dropna=False)
        .agg(
            scenarios=("true_p", "size"),
            min_valid_fit_rate=("fit_valid_rate", "min"),
            min_coverage_a=("coverage_a", "min"),
            max_coverage_a=("coverage_a", "max"),
            min_coverage_p=("coverage_p", "min"),
            max_coverage_p=("coverage_p", "max"),
            max_abs_bias_a=("bias_a", lambda x: float(np.max(np.abs(x)))),
            max_abs_bias_p=("bias_p", lambda x: float(np.max(np.abs(x)))),
        )
        .reset_index()
    )
    robustness.to_csv(output_directory / "table_robustness.csv", index=False)
    created.append("table_robustness.csv")
    return created


def self_check() -> None:
    for a, p in ((-1.0, 0.1), (-1.0, 0.6), (-2.0, 0.3)):
        if not np.isclose(momb_survival(0.0, a, p), 1.0, atol=1e-12):
            raise AssertionError("S(0) != 1")
        if not np.isclose(momb_survival(np.inf, a, p), p, atol=1e-12):
            raise AssertionError("S(infinity) != p")
        integral = quad(lambda value: float(momb_density(value, a, p)), 0.0, np.inf)[0]
        if not np.isclose(integral, 1.0 - p, rtol=2e-8, atol=2e-8):
            raise AssertionError("Integral da densidade diferente de 1-p")
        for conditional_level in (0.1, 0.5, 0.9):
            q = p + (1.0 - p) * conditional_level
            f_mb = survival_level_to_maxwell_cdf(q, p)
            t = maxwell.ppf(f_mb, scale=-a)
            if not np.isclose(momb_survival(t, a, p), q, rtol=2e-10, atol=2e-10):
                raise AssertionError("Inversao da sobrevivencia inconsistente")
        for target in (0.1, 0.3):
            tau = uniform_censoring_cutoff(a, p, target)
            achieved = quad(
                lambda value: float(conditional_survival(value, a, p)),
                0.0,
                tau,
                epsabs=1e-9,
                epsrel=1e-9,
            )[0] / tau
            if not np.isclose(achieved, target, rtol=2e-7, atol=2e-7):
                raise AssertionError("Calibracao da censura uniforme inconsistente")

    rng = np.random.default_rng(81173)
    scenario = Scenario(-1.0, 0.3, 250, 0.2)
    observed_time, delta, _ = simulate_cohort(rng, scenario)
    eta = np.array([np.log(1.1), logit(0.35)])
    analytic = nll_and_gradient(eta, observed_time, delta)[1]
    numeric = approx_fprime(eta, lambda value: nll_and_gradient(value, observed_time, delta)[0], 1e-6)
    if not np.allclose(analytic, numeric, rtol=2e-4, atol=2e-4):
        raise AssertionError(f"Gradiente inconsistente: analytic={analytic}, numeric={numeric}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[20, 50, 100, 250, 500, 1000])
    parser.add_argument("--a-values", type=float, nargs="+", default=[-1.0, -2.0])
    parser.add_argument("--p-values", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--susceptible-censoring", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument(
        "--mechanisms", nargs="+", choices=["random_uniform", "administrative"],
        default=["random_uniform"],
        help="Mecanismo de censura; o padrão final é independente uniforme.",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--gtol", type=float, default=1e-7)
    parser.add_argument(
        "--start-profile",
        choices=["full", "reduced"],
        default="full",
        help="Perfil de pontos iniciais: full (8) ou reduced (4).",
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("resultados_simulacao_censurada"))
    parser.add_argument("--resume", action="store_true", help="Retoma lotes concluídos no diretório existente.")
    parser.add_argument("--quick", action="store_true", help="Executa apenas um smoke test curto.")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.replications <= 1:
        raise ValueError("Use pelo menos duas replicacoes.")
    if any(n < 20 for n in args.sample_sizes):
        raise ValueError("Todos os tamanhos amostrais devem ser >= 20.")
    if any(a >= 0 for a in args.a_values):
        raise ValueError("Todos os valores de a devem ser negativos.")
    if any(not 0.0 < p < 1.0 for p in args.p_values):
        raise ValueError("Todos os valores de p devem estar em (0,1).")
    if any(not 0.0 <= c < 1.0 for c in args.susceptible_censoring):
        raise ValueError("A censura entre suscetiveis deve estar em [0,1).")
    if not args.mechanisms:
        raise ValueError("Informe pelo menos um mecanismo de censura.")
    if args.batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(master_seed: int, task_index: int) -> int:
    """Semente independente por tarefa, estável entre lotes e máquinas."""
    sequence = np.random.SeedSequence([int(master_seed), int(task_index)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def write_checksums(output_directory: Path, filenames: Iterable[str]) -> None:
    lines = []
    for filename in sorted(set(filenames)):
        path = output_directory / filename
        if path.exists() and path.is_file():
            lines.append(f"{file_sha256(path)}  {filename}")
    (output_directory / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def scenario_manifest(scenarios: list[Scenario]) -> pd.DataFrame:
    rows = []
    for index, scenario in enumerate(scenarios, start=1):
        row = asdict(scenario)
        row.update(
            {
                "scenario_id": index,
                "true_b": scenario.true_b,
                "expected_total_censoring": scenario.expected_total_censoring,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_arguments()
    if args.quick:
        args.replications = 8
        args.sample_sizes = [50, 100]
        args.a_values = [-1.0]
        args.p_values = [0.1, 0.5]
        args.susceptible_censoring = [0.2]
        args.mechanisms = ["random_uniform"]
        args.batch_size = 16
    validate_arguments(args)
    self_check()

    output_directory = args.output_dir.resolve()
    if output_directory.exists() and any(output_directory.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Diretorio de saida nao esta vazio: {output_directory}. "
            "Escolha outro --output-dir ou use --resume para retomar lotes."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    scenarios: list[Scenario] = []
    for a, p, n, censoring, mechanism in product(
        args.a_values,
        args.p_values,
        args.sample_sizes,
        args.susceptible_censoring,
        args.mechanisms,
    ):
        censoring_parameter = (
            uniform_censoring_cutoff(a, p, censoring)
            if mechanism == "random_uniform"
            else administrative_cutoff(a, p, censoring)
        )
        scenarios.append(
            Scenario(a, p, n, censoring, mechanism, censoring_parameter)
        )

    manifest = scenario_manifest(scenarios)
    manifest.to_csv(output_directory / "scenario_manifest.csv", index=False)
    number_of_tasks = len(scenarios) * args.replications
    script_path = Path(__file__).resolve()
    config_path = output_directory / "config.json"
    progress_path = output_directory / "progress.json"
    raw_path = output_directory / "replications.csv"
    config = {
        "experiment": "MOMB defective cure model — final reproducible Monte Carlo protocol",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": args.seed,
        "replications_per_scenario": args.replications,
        "sample_sizes": args.sample_sizes,
        "a_values": args.a_values,
        "p_values": args.p_values,
        "b_values": [float(b_from_p(p)) for p in args.p_values],
        "susceptible_censoring_targets": args.susceptible_censoring,
        "mechanisms": args.mechanisms,
        "number_of_scenarios": len(scenarios),
        "number_of_attempted_fits": number_of_tasks,
        "optimizer": "BFGS, data-derived truth-free starts",
        "start_profile": args.start_profile,
        "starts_per_replication": 8 if args.start_profile == "full" else 4,
        "uncertainty": "inverse central-difference observed Hessian in (log(-a), logit(p))",
        "confidence_intervals": "Wald in transformed space, back-transformed to valid parameter domains",
        "failure_policy": "retain and report every failed replication; never regenerate failures",
        "censoring_design": (
            "independent C~Uniform(0,tau), tau calibrated by quadrature to target "
            "P(C<T | susceptible); administrative is optional sensitivity"
        ),
        "maxiter": args.maxiter,
        "gtol": args.gtol,
        "jobs": args.jobs,
        "batch_size": args.batch_size,
        "parallel_backend": "threading",
        "package_versions": package_versions(),
        "script_sha256": file_sha256(script_path),
        "command": sys.argv,
    }

    if args.resume:
        if not config_path.exists() or not progress_path.exists() or not raw_path.exists():
            raise FileNotFoundError("--resume requer config.json, progress.json e replications.csv.")
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed_tasks = int(progress.get("completed_tasks", 0))
        for key in ("master_seed", "number_of_attempted_fits", "start_profile"):
            if old_config.get(key) != config.get(key):
                raise ValueError(f"Configuração incompatível para retomada: {key}.")
        if old_config.get("script_sha256") != config.get("script_sha256"):
            if completed_tasks < number_of_tasks:
                raise ValueError("O script mudou antes do término; não é seguro retomar os ajustes.")
            print("Aviso: script mudou após todos os ajustes; apenas os produtos serão regenerados.")
    else:
        completed_tasks = 0
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        progress_path.write_text(
            json.dumps({"completed_tasks": 0, "total_tasks": number_of_tasks}, indent=2),
            encoding="utf-8",
        )

    if completed_tasks > number_of_tasks:
        raise ValueError("progress.json indica mais tarefas concluídas do que o protocolo.")

    print(
        f"Self-checks passed. Running {number_of_tasks} fits across {len(scenarios)} scenarios "
        f"in batches of {args.batch_size}. Already completed: {completed_tasks}."
    )
    start_time = time.perf_counter()
    while completed_tasks < number_of_tasks:
        batch = []
        for task_index in range(completed_tasks, min(number_of_tasks, completed_tasks + args.batch_size)):
            scenario_index = task_index // args.replications
            replication_index = task_index % args.replications + 1
            batch.append(
                (
                    scenarios[scenario_index],
                    replication_index,
                    deterministic_seed(args.seed, task_index),
                )
            )
        rows = Parallel(n_jobs=args.jobs, backend="threading", verbose=0)(
                delayed(run_replication)(
                    scenario_item,
                    replication_item,
                    seed,
                    args.maxiter,
                    args.gtol,
                    args.start_profile,
                )
            for scenario_item, replication_item, seed in batch
        )
        batch_results = pd.DataFrame(rows)
        for required in (
            "optimizer_success", "gradient_inf_norm", "hessian_condition_number",
            "a_hat", "b_hat", "p_hat", "se_a", "se_b", "se_p",
            "a_ci_lower", "a_ci_upper", "b_ci_lower", "b_ci_upper",
            "p_ci_lower", "p_ci_upper",
        ):
            if required not in batch_results:
                batch_results[required] = np.nan
        batch_results["fit_valid"] = batch_results["fit_valid"].fillna(False).astype(bool)
        batch_results["optimizer_success"] = batch_results["optimizer_success"].fillna(False).astype(bool)
        batch_results = add_performance_columns(batch_results)
        batch_results.to_csv(
            raw_path,
            mode="a",
            header=not raw_path.exists() or raw_path.stat().st_size == 0,
            index=False,
        )
        completed_tasks += len(batch_results)
        progress_path.write_text(
            json.dumps(
                {
                    "completed_tasks": completed_tasks,
                    "total_tasks": number_of_tasks,
                    "last_batch_rows": len(batch_results),
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"Progress: {completed_tasks}/{number_of_tasks} "
            f"({100 * completed_tasks / number_of_tasks:.1f}%)"
        )

    elapsed_seconds = time.perf_counter() - start_time
    results = pd.read_csv(raw_path)
    for column in ("fit_valid", "optimizer_success"):
        results[column] = results[column].astype(bool)
    summary = summarize_results(results)
    coverage_assessment = coverage_assessment_table(summary)
    failures = results[~results["fit_valid"]].copy()
    summary.to_csv(output_directory / "summary.csv", index=False)
    coverage_assessment.to_csv(output_directory / "coverage_assessment.csv", index=False)
    failures.to_csv(output_directory / "failures.csv", index=False)
    figures = plot_summary(summary, output_directory)
    figures.extend(plot_coverage_diagnostics(summary, output_directory))
    figures.extend(plot_quality_diagnostics(summary, output_directory))
    tables = write_article_tables(summary, output_directory)

    config.update(
        {
            "elapsed_seconds": elapsed_seconds,
            "valid_fits": int(results["fit_valid"].sum()),
            "failed_fits": int((~results["fit_valid"]).sum()),
            "created_figures": figures,
            "created_tables": tables,
            "completed_tasks": int(completed_tasks),
        }
    )
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    write_checksums(
        output_directory,
        [
            "config.json", "scenario_manifest.csv", "replications.csv", "summary.csv",
            "coverage_assessment.csv", "failures.csv", *figures, *tables,
        ],
    )

    print(
        f"Completed in {elapsed_seconds:.1f}s: {int(results['fit_valid'].sum())}/{len(results)} valid fits.\n"
        f"Outputs: {output_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

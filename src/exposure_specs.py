from __future__ import annotations

PRIMARY_EXPOSURE_SPECS = [
    "exposure_cov_hybrid",
    "exposure_exp_hybrid",
]

COVERAGE_EXPOSURE_SPECS = [
    "exposure_cov_hybrid",
    "exposure_cov_sim",
    "exposure_cov_admin",
]

BENEFIT_EXPOSURE_SPECS = [
    "exposure_exp_hybrid",
    "exposure_exp_sim",
    "exposure_exp_admin",
]

EXPOSURE_LABELS = {
    "exposure_cov_hybrid": "Hybrid coverage exposure",
    "exposure_exp_hybrid": "Hybrid average-benefit exposure",
    "exposure_cov_sim": "Simulated coverage exposure",
    "exposure_exp_sim": "Simulated average-benefit exposure",
    "exposure_cov_admin": "Administrative coverage exposure",
    "exposure_exp_admin": "Administrative average-benefit exposure",
}

RAW_EXPOSURE_MAP = {
    "exposure_cov_hybrid": "delta_cov_hybrid",
    "exposure_exp_hybrid": "delta_benefit_hybrid",
    "exposure_cov_sim": "delta_cov_sim",
    "exposure_exp_sim": "delta_benefit_sim",
    "exposure_cov_admin": "level_cov_admin",
    "exposure_exp_admin": "level_benefit_admin",
}

RAW_PRIMARY_EXPOSURES = [
    RAW_EXPOSURE_MAP[spec] for spec in PRIMARY_EXPOSURE_SPECS
]

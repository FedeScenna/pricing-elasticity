"""Bayesian price elasticity of demand estimation with PyMC."""

from pricing_elasticity.data import MockData, generate_aids_data, generate_blp_data, generate_mock_data
from pricing_elasticity.features import blp_instruments, competitor_price_columns, fourier_features
from pricing_elasticity.models import (
    AIDSDemandModel,
    BLPDemandModel,
    DemandModel,
    LogLogDemandModel,
    build_log_log_model,
    fit_log_log_model,
    summarize_cross_elasticities,
    summarize_elasticities,
)

__all__ = [
    "AIDSDemandModel",
    "BLPDemandModel",
    "DemandModel",
    "LogLogDemandModel",
    "MockData",
    "blp_instruments",
    "build_log_log_model",
    "competitor_price_columns",
    "fit_log_log_model",
    "fourier_features",
    "generate_aids_data",
    "generate_blp_data",
    "generate_mock_data",
    "summarize_cross_elasticities",
    "summarize_elasticities",
]

__version__ = "0.2.0"

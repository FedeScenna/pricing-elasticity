"""Synthetic data generators for each demand model."""

from pricing_elasticity.data._common import MockData
from pricing_elasticity.data.aids import generate_aids_data
from pricing_elasticity.data.blp import generate_blp_data
from pricing_elasticity.data.loglog import generate_mock_data

__all__ = ["MockData", "generate_aids_data", "generate_blp_data", "generate_mock_data"]

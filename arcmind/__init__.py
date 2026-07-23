"""
ArcMind: a causal dual-rate sequence backbone for streaming sensor policies.
"""

__version__ = "0.2.0"

from arcmind.config.defaults import ArcMindConfig
from arcmind.models.arcmind_model import ArcMindModel

__all__ = ["ArcMindConfig", "ArcMindModel", "__version__"]

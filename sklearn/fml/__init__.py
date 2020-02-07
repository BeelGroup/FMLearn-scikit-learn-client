"""
The :mod:`sklearn.fml` module implements Federated Meta Learning
"""

from .fml_client import FMLClient
from .constants import URI

__all__ = ['FMLClient',
            'URI']

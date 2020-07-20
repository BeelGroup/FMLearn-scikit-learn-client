"""
The :mod:`sklearn.fmlearn` module implements Federated Meta Learning
"""

from .fml_client import FMLClient
from .constants import URI
from .metafeatures import MetaFeatures

__all__ = ['FMLClient',
            'URI',
            'MetaFeatures']

"""
The :mod:`nilearn.decomposition` module includes a subject level
variant of the ICA called Canonical ICA.
"""
from .canica import CanICA
from .dict_learning import DictLearning
from .multi_pca import MultiPCA
__all__ = ['CanICA', 'DictLearning','MultiPCA']
